using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;
using System.Runtime.InteropServices;
using Xunit;

public sealed class CudaDrnRetentionFloorTests
{
    [Fact]
    public void NativeAbiProvidesFloorAwareKernelsAndRetainsLegacyExports()
    {
        Assert.True(CudaNativeGateway.AbiVersion.Minor >= CudaAbiVersion.DrnRetentionFloorMinor);
        nint library = NativeLibrary.Load(Path.Combine(AppContext.BaseDirectory, CudaNativeGateway.LibraryName));
        try
        {
            foreach (string name in new[]
            {
                "nntrain_drn_chunk_forward", "nntrain_drn_chunk_backward",
                "nntrain_drn_chunk_parallel_forward", "nntrain_drn_backward_prepared",
            })
            {
                Assert.True(NativeLibrary.TryGetExport(library, name, out _), $"Missing legacy export: {name}");
                Assert.True(NativeLibrary.TryGetExport(library, name + "_with_floor", out _), $"Missing floor-aware export: {name}");
            }
        }
        finally { NativeLibrary.Free(library); }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 2, 7, 3, 2)]
    [InlineData(TensorPrecisionMode.Mix16_32, 2, 7, 3, 2)]
    [InlineData(TensorPrecisionMode.Mix8_32, 2, 7, 3, 2)]
    [InlineData(TensorPrecisionMode.Float32, 8, 33, 16, 16)]
    [InlineData(TensorPrecisionMode.Mix16_32, 8, 33, 16, 16)]
    [InlineData(TensorPrecisionMode.Mix8_32, 8, 33, 16, 16)]
    [InlineData(TensorPrecisionMode.Float32, 2, 65, 32, 32)]
    [InlineData(TensorPrecisionMode.Mix16_32, 2, 65, 32, 32)]
    [InlineData(TensorPrecisionMode.Mix8_32, 2, 65, 32, 32)]
    public void ForwardAndGradientHonorFloorAgainstDecodedCpuReference(
        TensorPrecisionMode mode, int batch, int sequence, int key, int value)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        int width = 2 * key + 3 * value;
        float[] source = Projections(batch * sequence, key, value, UsesTensorCore(mode, key, value));
        float[] upstream = Enumerable.Range(0, batch * sequence * value)
            .Select(i => .01f * (1 + i % 3)).ToArray();
        int[] shape = [batch, sequence, width];
        float[] decoded;
        using (var cpu = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu)))
            decoded = DecodedInput(source, shape, mode);

        float[]? zeroFloorOutput = null;
        float[]? zeroFloorGradient = null;
        foreach (float floor in new[] { 0f, .5f, .99f })
        {
            (float[] output, float[] gradient) expected;
            using (var cpu = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu)))
            using (var precision = TensorExecutionContext.PushPrecisionPolicy(PrecisionPolicy.Float32))
            {
                var input = new Tensor(decoded.ToArray(), shape);
                Tensor output = input.ForgetMemoryDRN(key, value, floor);
                output.Backward(upstream);
                expected = (QuantizeOutput(output.Data.ToArray(), mode), input.Grad.ToArray());
                if (UsesTensorCore(mode, key, value))
                {
                    var basis = TensorCoreBasisReference(decoded, batch, sequence, key, value, floor, upstream);
                    expected = (QuantizeOutput(basis.Output, mode), basis.Gradient);
                }
            }

            using var session = Session(mode);
            using var scope = session.Enter();
            using var dispatch = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
            Tensor projected = Input(source, shape, mode);
            Tensor? actual = null;
            try
            {
                projected.to(new TorchDevice(TensorDevice.Cuda, 0));
                actual = projected.ForgetMemoryDRN(key, value, floor);
                float[] output = actual.Data.ToArray();
                actual.BackwardAndRelease(upstream);
                float[] gradient = projected.Grad.ToArray();
                // The independent TC reference rounds state operands, not the
                // FP32 accumulator. Existing comparison tolerances are retained.
                Compare(expected.output, output, 3e-5f);
                Compare(expected.gradient, gradient, 6e-5f);
                int gateOffset = 2 * key + value;
                Assert.Contains(Enumerable.Range(0, batch * sequence)
                    .SelectMany(t => gradient.AsSpan(t * width + gateOffset, value).ToArray()),
                    x => MathF.Abs(x) > 1e-7f);
                if (floor == 0f)
                {
                    zeroFloorOutput = output;
                    zeroFloorGradient = gradient;
                }
                else
                {
                    Assert.True(output.Zip(zeroFloorOutput!, (a, b) => MathF.Abs(a - b)).Max() > .001f,
                        "Changing retentionFloor must change the recurrent output.");
                    Assert.True(gradient.Zip(zeroFloorGradient!, (a, b) => MathF.Abs(a - b)).Max() > 1e-5f,
                        "Changing retentionFloor must change backward, including its gate derivative.");
                }
            }
            finally
            {
                actual?.InvalidateCudaBuffers();
                projected.InvalidateCudaBuffers();
            }
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 3, 2)]
    [InlineData(TensorPrecisionMode.Mix16_32, 16, 16)]
    [InlineData(TensorPrecisionMode.Mix8_32, 32, 32)]
    public void ResidentContinuationHonorsFloorAcrossCalls(
        TensorPrecisionMode mode, int key, int value)
    {
        Assert.SkipWhen(!Tensor.IsCudaAvailable(), "CUDA is unavailable.");
        int width = 2 * key + 3 * value;
        const int firstLength = 3, secondLength = 4;
        float[] firstSource = Projections(firstLength, key, value, UsesTensorCore(mode, key, value));
        float[] secondSource = Projections(secondLength, key, value, UsesTensorCore(mode, key, value));
        float[] initial = Enumerable.Range(0, key * value)
            .Select(i => (i % 5 - 2) * .03125f).ToArray();
        float[]? zeroFloorState = null;
        foreach (float floor in new[] { 0f, .5f, .99f })
        {
            float[] firstExpected, secondExpected, expectedState = initial.ToArray();
            using (var cpu = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu)))
            using (var precision = TensorExecutionContext.PushPrecisionPolicy(PrecisionPolicy.Float32))
            using (AutogradContext.NoGrad())
            {
                var first = new Tensor(DecodedInput(firstSource, [1, firstLength, width], mode),
                    [1, firstLength, width]);
                var second = new Tensor(DecodedInput(secondSource, [1, secondLength, width], mode),
                    [1, secondLength, width]);
                firstExpected = QuantizeOutput(first.ForgetMemoryDRNContinue(key, value, floor, expectedState).Data.ToArray(), mode);
                secondExpected = QuantizeOutput(second.ForgetMemoryDRNContinue(key, value, floor, expectedState).Data.ToArray(), mode);
                if (UsesTensorCore(mode, key, value))
                {
                    expectedState = initial.ToArray();
                    firstExpected = QuantizeOutput(TensorCoreBasisForward(first.Data.ToArray(),
                        firstLength, key, value, floor, expectedState), mode);
                    secondExpected = QuantizeOutput(TensorCoreBasisForward(second.Data.ToArray(),
                        secondLength, key, value, floor, expectedState), mode);
                }
            }

            using var session = Session(mode);
            using var scope = session.Enter();
            using var dispatch = CudaDispatchPolicy.Push(CudaDispatchPolicy.Defaults);
            using var noGrad = AutogradContext.NoGrad();
            using var inference = CudaInferenceScope.Begin(resetPool: true, clearPoolOnDispose: true);
            using var state = new ForgetMemoryRecurrentMemory(initial.Length);
            initial.CopyTo(state.HostForCpuMutation(), 0);
            state.MarkHostMutated();
            Tensor firstInput = Input(firstSource, [1, firstLength, width], mode);
            Tensor secondInput = Input(secondSource, [1, secondLength, width], mode);
            try
            {
                firstInput.to(new TorchDevice(TensorDevice.Cuda, 0));
                secondInput.to(new TorchDevice(TensorDevice.Cuda, 0));
                Tensor first = firstInput.ForgetMemoryDRNContinue(key, value, floor, state);
                Tensor second = secondInput.ForgetMemoryDRNContinue(key, value, floor, state);
                Compare(firstExpected, first.Data, 3e-5f);
                Compare(secondExpected, second.Data, 3e-5f);
                float[] actualState = state.HostSnapshot().ToArray();
                Compare(expectedState, actualState, 3e-5f);
                if (floor == 0f) zeroFloorState = actualState;
                else Assert.True(actualState.Zip(zeroFloorState!, (a, b) => MathF.Abs(a - b)).Max() > .001f);
            }
            finally
            {
                firstInput.InvalidateCudaBuffers();
                secondInput.InvalidateCudaBuffers();
            }
        }
    }

    private static ExecutionSession Session(TensorPrecisionMode mode)
        => new(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet([0]),
            Precision = mode switch
            {
                TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
                TensorPrecisionMode.Mix16_32 => PrecisionPolicy.Mix16_32,
                TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
                _ => throw new ArgumentOutOfRangeException(nameof(mode)),
            },
        }, [CudaExecutionLaneFactory.Create(0)]);

    private static Tensor Input(float[] source, int[] shape, TensorPrecisionMode mode)
        => mode == TensorPrecisionMode.Mix8_32
            ? Tensor.FromBfp8(source.ToArray(), shape, Bfp8QuantizationDescriptor.Mix8_32)
            : new Tensor(source.ToArray(), shape,
                dtype: mode == TensorPrecisionMode.Float32 ? TensorDType.Float32 : TensorDType.BFloat16);

    private static float[] QuantizeOutput(float[] source, TensorPrecisionMode mode)
    {
        if (mode == TensorPrecisionMode.Float32) return source;
        float[] bf16 = source.Select(TensorStorageCodec.RoundToBFloat16).ToArray();
        return mode == TensorPrecisionMode.Mix16_32 ? bf16
            : Tensor.FromBfp8(bf16, [bf16.Length], Bfp8QuantizationDescriptor.Mix8_32).Data.ToArray();
    }

    private static float[] DecodedInput(float[] source, int[] shape, TensorPrecisionMode mode)
    {
        float[] decoded = Input(source, shape, mode).Data.ToArray();
        // The CUDA mix8 path decodes block-scaled weights/activations to BF16
        // operands before the recurrence; compare the same operands on CPU.
        return mode == TensorPrecisionMode.Mix8_32
            ? decoded.Select(TensorStorageCodec.RoundToBFloat16).ToArray() : decoded;
    }

    private static bool UsesTensorCore(TensorPrecisionMode mode, int key, int value)
        => mode != TensorPrecisionMode.Float32 && key % 16 == 0 && value % 16 == 0;

    private static (float[] Output, float[] Gradient) TensorCoreBasisReference(
        float[] projected, int batch, int sequence, int key, int value, float floor, float[] upstream)
    {
        int width = 2 * key + 3 * value;
        var output = new float[batch * sequence * value];
        var gradient = new float[projected.Length];
        for (int b = 0; b < batch; b++)
        {
            var history = new float[sequence * value];
            float[] projection = projected.AsSpan(b * sequence * width, sequence * width).ToArray();
            TensorCoreBasisForward(projection, sequence, key, value, floor,
                new float[key * value], history).CopyTo(output, b * sequence * value);
            for (int v = 0; v < value; v++)
            {
                float adjoint = 0f;
                for (int t = sequence - 1; t >= 0; t--)
                {
                    int p = (b * sequence + t) * width + 2 * key + v;
                    float m = history[t * value + v];
                    float z = MathF.Tanh(projected[p]);
                    float gate = Sigmoid(projected[p + value]);
                    float beta = Sigmoid(projected[p + 2 * value]);
                    float retention = floor + (1f - floor) * gate;
                    gradient[p] = adjoint * beta * (1f - z * z);
                    gradient[p + value] = adjoint * m * (1f - floor) * gate * (1f - gate);
                    gradient[p + 2 * value] = adjoint * (z - m) * beta * (1f - beta);
                    adjoint = upstream[(b * sequence + t) * value + v] + adjoint * (retention - beta);
                }
            }
        }
        return (output, gradient);
    }

    // For the exact unit-basis Q/K fixture, the matrix recurrence reduces to
    // independent scalar rows. TC reads use BF16 state operands while retained
    // state, gate/value arithmetic and backward accumulation stay FP32. The
    // current backward contract differentiates the FP32 recurrence (STE), so
    // dBeta uses unrounded previous state just like the non-TC CUDA reference.
    private static float[] TensorCoreBasisForward(float[] projected, int sequence,
        int key, int value, float floor, float[] state, float[]? history = null)
    {
        int width = 2 * key + 3 * value;
        var output = new float[sequence * value];
        for (int t = 0; t < sequence; t++)
        {
            int p = t * width;
            Assert.Equal(1f, MathF.Tanh(projected[p]));
            Assert.Equal(1f, MathF.Tanh(projected[p + key]));
            for (int k = 1; k < key; k++)
            {
                Assert.Equal(0f, projected[p + k]);
                Assert.Equal(0f, projected[p + key + k]);
            }
            for (int v = 0; v < value; v++)
            {
                int row = v * key, offset = p + 2 * key + v;
                float previous = state[row];
                if (history is not null) history[t * value + v] = previous;
                float operand = TensorStorageCodec.RoundToBFloat16(previous);
                output[t * value + v] = operand;
                float gate = 1f / (1f + MathF.Exp(-projected[offset + value]));
                float retention = floor + (1f - floor) * gate;
                float beta = 1f / (1f + MathF.Exp(-projected[offset + 2 * value]));
                float delta = beta * (MathF.Tanh(projected[offset]) - operand);
                state[row] = MathF.FusedMultiplyAdd(retention, previous, delta);
                for (int k = 1; k < key; k++) state[row + k] *= retention;
            }
        }
        return output;
    }

    private static float Sigmoid(float x)
    {
        if (x >= 0f) return 1f / (1f + MathF.Exp(-x));
        float e = MathF.Exp(x);
        return e / (1f + e);
    }

    private static float[] Projections(int tokens, int key, int value, bool exactNonlinear)
    {
        int width = 2 * key + 3 * value;
        var source = new float[tokens * width];
        for (int t = 0; t < tokens; t++)
        {
            int offset = t * width;
            source[offset] = source[offset + key] = exactNonlinear ? 20f : 8f;
            for (int v = 0; v < value; v++)
            {
                if (exactNonlinear)
                {
                    // tanh(0/±20)=0/±1 and sigmoid(0)=.5 are exactly
                    // representable CPU and CUDA values. This isolates floor
                    // arithmetic and TC state rounding from transcendentals'
                    // last-bit differences without relaxing comparisons.
                    source[offset + 2 * key + v] = ((t + v) % 3 - 1) * 20f;
                    continue;
                }
                // Vary the stream so CPU/CUDA exp/tanh last-bit differences
                // cannot select different phases of a quantized fixed-point
                // limit cycle after dozens of otherwise identical tokens.
                source[offset + 2 * key + v] = .25f + (v % 4) * .0625f + (t % 7 - 3) * .015625f;
                source[offset + 2 * key + value + v] = -.75f + (t % 5 - 2) * .0625f;
                source[offset + 2 * key + 2 * value + v] = -.25f + (t % 3) * .0625f;
            }
        }
        return source;
    }

    private static void Compare(IReadOnlyList<float> expected, IReadOnlyList<float> actual, float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"Nonfinite value at {i}.");
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tolerance,
                $"index={i}, expected={expected[i]:G9}, actual={actual[i]:G9}, tolerance={tolerance:G9}");
        }
    }
}
