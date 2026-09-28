using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingDeltaTests
{
    [Theory]
    [InlineData(2, 4, 7, 4)]
    [InlineData(1, 2, 128, 4)]
    [InlineData(2, 4, 1, 1)]
    public void SequenceForwardMatchesCpuAndRetainsNoRecurrentTape(int keys, int heads, int width, int kernel)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        var data = new Data(keys, heads, width, kernel, 4);
        using var buffers = new Inputs(lane, data);
        long baseline = lane.AllocatedBytes, uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        using (var training = buffers.Forward(lane, data))
        {
            Assert.Equal(uploaded, lane.H2DBytes);
            Assert.Equal(downloaded, lane.D2HBytes);
            // Only two channel activations and two value activations survive forward.
            Assert.Equal(baseline + (long)data.Sequence * (2 * data.Channels + 2 * data.Values) * sizeof(float), lane.AllocatedBytes);
            AssertClose(data.Reference(), Read(lane, training.Output, data.Sequence * data.Values), 5e-5f, 2e-4f);
        }
        Assert.Equal(baseline, lane.AllocatedBytes);
        Assert.True(buffers.Qkv.IsAlive && buffers.Gate.IsAlive && buffers.Conv.IsAlive);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExactBpttMatchesFiniteDifferencesAndAccumulates(bool lastTokenLossOnly)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        var data = new Data(2, 4, 3, 3, 4);
        using var buffers = new Inputs(lane, data);
        float[] dy = Enumerable.Range(0, data.Sequence * data.Values)
            .Select(i => lastTokenLossOnly && i < (data.Sequence - 1) * data.Values ? 0f : MathF.Sin(i * 0.71f + 0.3f)).ToArray();
        using ArcBuffer dyGpu = lane.Upload(dy);
        const float initial = 0.03125f;
        using ArcBuffer dqkv = lane.Upload(Enumerable.Repeat(initial, data.Qkv.Length).ToArray());
        using ArcBuffer dgate = lane.Upload(Enumerable.Repeat(initial, data.Gate.Length).ToArray());
        using ArcBuffer dalpha = lane.Upload(Enumerable.Repeat(initial, data.Alpha.Length).ToArray());
        using ArcBuffer dbeta = lane.Upload(Enumerable.Repeat(initial, data.Beta.Length).ToArray());
        using var training = buffers.Forward(lane, data);
        long bytes = lane.AllocatedBytes, uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        training.Backward(dyGpu, dqkv, dgate, dalpha, dbeta);
        Assert.Equal(bytes, lane.AllocatedBytes);
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded, lane.D2HBytes);
        float[][] gradients = [Read(lane, dqkv, data.Qkv.Length), Read(lane, dgate, data.Gate.Length),
            Read(lane, dalpha, data.Alpha.Length), Read(lane, dbeta, data.Beta.Length)];
        float[][] inputs = [data.Qkv, data.Gate, data.Alpha, data.Beta];
        string[] names = ["qkv", "gate", "alpha", "beta"];
        for (int group = 0; group < inputs.Length; ++group)
        for (int index = 0; index < inputs[group].Length; ++index)
        {
            const float step = 0.002f;
            float original = inputs[group][index];
            inputs[group][index] = original + step;
            double plus = Loss(data.Reference(), dy);
            inputs[group][index] = original - step;
            double minus = Loss(data.Reference(), dy);
            inputs[group][index] = original;
            float expected = (float)((plus - minus) / (2 * step));
            float actual = gradients[group][index] - initial;
            Assert.True(float.IsFinite(actual) && MathF.Abs(actual - expected) <= 0.003f + 0.008f * MathF.Abs(expected),
                $"{names[group]}[{index}] (last-only={lastTokenLossOnly}): numerical {expected:R}, GPU {actual:R}");
        }
        if (lastTokenLossOnly)
        {
            // Last-token loss must travel back through recurrent state and convolution.
            Assert.Contains(gradients[0].Take(data.Channels), g => MathF.Abs(g - initial) > 1e-4f);
            Assert.Contains(gradients[3].Take(data.Heads), g => MathF.Abs(g - initial) > 1e-4f);
            Assert.All(gradients[1].Take((data.Sequence - 1) * data.Values), g => Assert.Equal(initial, g));
        }
        training.Backward(dyGpu, dqkv, dgate, dalpha, dbeta);
        float[][] twice = [Read(lane, dqkv, data.Qkv.Length), Read(lane, dgate, data.Gate.Length),
            Read(lane, dalpha, data.Alpha.Length), Read(lane, dbeta, data.Beta.Length)];
        for (int group = 0; group < gradients.Length; ++group)
            AssertClose(gradients[group].Select(g => 2f * g - initial).ToArray(), twice[group], 2e-5f, 1e-5f);
        Assert.Equal(bytes, lane.AllocatedBytes);
    }

    [Theory]
    [InlineData(65, 63, 64)]
    [InlineData(129, 127, 128)]
    public void LastTokenGradientCrossesRecurrentChunkBoundary(int sequence, int previous, int lossTime)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        var data = new Data(1, 2, 3, 1, sequence);
        using var buffers = new Inputs(lane, data);
        var dy = new float[sequence * data.Values];
        for (int i = 0; i < data.Values; ++i)
            dy[lossTime * data.Values + i] = MathF.Sin(i * 0.67f + 0.4f);
        using ArcBuffer dyGpu = lane.Upload(dy);
        using ArcBuffer dqkv = lane.Upload(new float[data.Qkv.Length]);
        using ArcBuffer dgate = lane.Upload(new float[data.Gate.Length]);
        using ArcBuffer dalpha = lane.Upload(new float[data.Alpha.Length]);
        using ArcBuffer dbeta = lane.Upload(new float[data.Beta.Length]);
        using var training = buffers.Forward(lane, data);
        training.Backward(dyGpu, dqkv, dgate, dalpha, dbeta);

        float[] alphaGradient = Read(lane, dalpha, data.Alpha.Length);
        float[] betaGradient = Read(lane, dbeta, data.Beta.Length);
        float Numerical(float[] values, int index)
        {
            const float step = 0.002f;
            float original = values[index];
            values[index] = original + step;
            double plus = Loss(data.Reference(), dy);
            values[index] = original - step;
            double minus = Loss(data.Reference(), dy);
            values[index] = original;
            return (float)((plus - minus) / (2 * step));
        }
        int at = previous * data.Heads;
        float expectedAlpha = Numerical(data.Alpha, at);
        float expectedBeta = Numerical(data.Beta, at);
        Assert.True(MathF.Abs(expectedAlpha) + MathF.Abs(expectedBeta) > 1e-4f,
            "The boundary must carry a measurable recurrent dependency.");
        Assert.True(MathF.Abs(alphaGradient[at] - expectedAlpha) <= 0.001f + 0.01f * MathF.Abs(expectedAlpha),
            $"alpha across {previous}->{lossTime}: numerical {expectedAlpha:R}, GPU {alphaGradient[at]:R}");
        Assert.True(MathF.Abs(betaGradient[at] - expectedBeta) <= 0.001f + 0.01f * MathF.Abs(expectedBeta),
            $"beta across {previous}->{lossTime}: numerical {expectedBeta:R}, GPU {betaGradient[at]:R}");
    }

    [Theory]
    [InlineData(63, 2, 4, 32)]
    [InlineData(64, 2, 4, 32)]
    [InlineData(65, 2, 4, 32)]
    [InlineData(127, 2, 4, 32)]
    [InlineData(128, 2, 4, 32)]
    [InlineData(129, 2, 4, 32)]
    [InlineData(65, 16, 48, 128)]
    public void ForwardCapturedCheckpointsMatchSeparateReplay(int sequence, int keys, int heads, int width)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        var data = new Data(keys, heads, width, 3, sequence);
        int stateSize = data.Values * data.Width;
        int chunks = 1 + (sequence - 1) / 64;
        using ArcBuffer mixed = lane.Upload(data.Qkv);
        using ArcBuffer alpha = lane.Upload(data.Alpha);
        using ArcBuffer beta = lane.Upload(data.Beta);
        using ArcBuffer dt = lane.Upload(data.Dt);
        using ArcBuffer a = lane.Upload(data.A);
        using ArcBuffer state = lane.Allocate(stateSize);
        using ArcBuffer replayState = lane.Allocate(stateSize);
        using ArcBuffer captured = lane.Allocate(chunks * stateSize);
        using ArcBuffer replayed = lane.Allocate(chunks * stateSize);
        using ArcBuffer raw = lane.Allocate(sequence * data.Values);

        lane.Run("q35t_delta_recur", data.Values, 0,
            mixed, alpha, beta, dt, a, state, captured, raw,
            sequence, 64, data.Keys, data.Heads, data.Width, 0, 1);
        lane.Run("q35t_delta_checkpoints", data.Values, 0,
            mixed, alpha, beta, dt, a, replayState, replayed,
            sequence, 64, data.Keys, data.Heads, data.Width);
        AssertClose(Read(lane, replayed, chunks * stateSize),
            Read(lane, captured, chunks * stateSize), 1e-7f, 1e-7f);
    }

    [Fact]
    public void GateBackwardUsingForwardOutputMatchesDirectDerivative()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        const int sequence = 5, heads = 48, width = 128;
        const float eps = 1e-4f, initial = 0.03125f;
        var random = new Random(2735);
        float[] Values(int count) => Enumerable.Range(0, count)
            .Select(_ => (float)(random.NextDouble() * 2 - 1) * 0.6f).ToArray();
        float[] raw = Values(sequence * heads * width), gate = Values(raw.Length);
        float[] dy = Values(raw.Length), norm = Values(width).Select(x => x + 1f).ToArray();
        using ArcBuffer rawGpu = lane.Upload(raw), gateGpu = lane.Upload(gate);
        using ArcBuffer normGpu = lane.Upload(norm), dyGpu = lane.Upload(dy);
        using ArcBuffer outputGpu = lane.Allocate(raw.Length), drawGpu = lane.Allocate(raw.Length);
        using ArcBuffer dgateGpu = lane.Upload(Enumerable.Repeat(initial, raw.Length).ToArray());
        lane.Run("q35t_delta_gate", sequence * heads, 0,
            rawGpu, gateGpu, normGpu, outputGpu, sequence, heads, width, eps);
        lane.Run("q35t_delta_gate_backward", sequence * heads, 0,
            rawGpu, gateGpu, normGpu, outputGpu, dyGpu, drawGpu, dgateGpu,
            sequence, heads, width, eps);

        var expectedDraw = new float[raw.Length];
        var expectedGate = new float[raw.Length];
        for (int head = 0; head < sequence * heads; head++)
        {
            int offset = head * width;
            float ss = 0f, dot = 0f;
            for (int v = 0; v < width; v++)
            {
                int i = offset + v;
                float s = Sigmoid(gate[i]);
                ss += raw[i] * raw[i];
                dot += raw[i] * dy[i] * norm[v] * (gate[i] * s);
            }
            float inv = 1f / MathF.Sqrt(ss / width + eps);
            float correction = dot * inv * inv / width;
            for (int v = 0; v < width; v++)
            {
                int i = offset + v;
                float s = Sigmoid(gate[i]);
                expectedDraw[i] = inv * (dy[i] * norm[v] * (gate[i] * s) - raw[i] * correction);
                expectedGate[i] = initial + dy[i] * raw[i] * inv * norm[v]
                    * (s * (1f + gate[i] * (1f - s)));
            }
        }
        AssertClose(expectedDraw, Read(lane, drawGpu, raw.Length), 2e-5f, 2e-4f);
        AssertClose(expectedGate, Read(lane, dgateGpu, raw.Length), 2e-5f, 2e-4f);

        static float Sigmoid(float x) => x >= 0f
            ? 1f / (1f + MathF.Exp(-x))
            : MathF.Exp(x) / (1f + MathF.Exp(x));
    }

    [Theory]
    [InlineData(65, 2, 4, 32)]
    [InlineData(129, 2, 4, 32)]
    [InlineData(65, 16, 48, 128)]
    public void SkippingUnusedForwardCheckpointsPreservesOutputAndBackward(
        int sequence, int keys, int heads, int width)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        var data = new Data(keys, heads, width, 3, sequence);
        using var buffers = new Inputs(lane, data);
        float[] dy = Enumerable.Range(0, sequence * data.Values)
            .Select(i => MathF.Sin(i * 0.071f + 0.3f)).ToArray();
        using ArcBuffer dyGpu = lane.Upload(dy);
        long baseline = lane.AllocatedBytes;

        (long Live, float[][] Values) Run(bool capture)
        {
            using ArcBuffer dqkv = lane.Upload(new float[data.Qkv.Length]);
            using ArcBuffer dgate = lane.Upload(new float[data.Gate.Length]);
            using ArcBuffer dalpha = lane.Upload(new float[data.Alpha.Length]);
            using ArcBuffer dbeta = lane.Upload(new float[data.Beta.Length]);
            using var training = buffers.Forward(lane, data, capture);
            long live = lane.AllocatedBytes;
            float[] output = Read(lane, training.Output, sequence * data.Values);
            training.Backward(dyGpu, dqkv, dgate, dalpha, dbeta);
            return (live, [output, Read(lane, dqkv, data.Qkv.Length),
                Read(lane, dgate, data.Gate.Length), Read(lane, dalpha, data.Alpha.Length),
                Read(lane, dbeta, data.Beta.Length)]);
        }

        var captured = Run(true);
        var skipped = Run(false);
        long checkpointBytes = (long)(1 + (sequence - 1) / 64) * data.Values * data.Width * sizeof(float);
        Assert.Equal(checkpointBytes, captured.Live - skipped.Live);
        for (int i = 0; i < captured.Values.Length; i++)
            AssertClose(captured.Values[i], skipped.Values[i], 3e-5f, 3e-5f);
        Assert.Equal(baseline, lane.AllocatedBytes);
    }

    private static ArcExecutionLane CreateLane() => new(0, new()
        { Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true });

    [Theory]
    [InlineData(2, 6, 7)]
    [InlineData(16, 48, 128)]
    [InlineData(2, 6, 129)]
    public void CooperativeQkBackwardMatchesSerialForGroupedHeadsAndPartialValueTiles(int keys, int heads, int width)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = CreateLane();
        const int time = 1, sequence = 3;
        int values = heads * width, channels = (2 * keys + heads) * width;
        var random = new Random(821 + width);
        float[] Random(int count) => Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        using ArcBuffer alpha = lane.Upload(Random(sequence * heads));
        using ArcBuffer dt = lane.Upload(Random(heads));
        using ArcBuffer a = lane.Upload(Random(heads).Select(x => -MathF.Abs(x)).ToArray());
        using ArcBuffer states = lane.Upload(Random((sequence + 1) * values * width));
        using ArcBuffer draw = lane.Upload(Random(sequence * values));
        using ArcBuffer adj = lane.Upload(Random(values * width));
        using ArcBuffer scratch = lane.Upload(Random(values * 4));
        // Untouched tokens and value gradients must survive both paths unchanged.
        float[] initial = Random(sequence * channels);
        using ArcBuffer serial = lane.Upload(initial);
        using ArcBuffer cooperative = lane.Upload(initial);
        lane.Run("q35t_delta_qk_backward_step", keys * width, 0,
            alpha, dt, a, states, draw, adj, scratch, serial, time, 0, keys, heads, width);
        lane.Run("q35t_delta_qk_backward_step_cooperative", (long)keys * width * 32, 32,
            alpha, dt, a, states, draw, adj, scratch, cooperative, time, 0, keys, heads, width);
        AssertClose(Read(lane, serial, initial.Length), Read(lane, cooperative, initial.Length), 3e-5f, 3e-5f);
    }

    [Theory]
    [InlineData(65, 2, 4, 32)]
    [InlineData(129, 2, 4, 32)]
    [InlineData(5, 16, 48, 128)]
    public void PersistentChunkBackwardMatchesPerTokenOracle(int sequence, int keys, int heads, int width)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true,
            Qwen35CooperativeDelta = true
        });
        var data = new Data(keys, heads, width, 3, sequence);
        using var buffers = new Inputs(lane, data);
        float[] dy = Enumerable.Range(0, sequence * data.Values)
            .Select(i => MathF.Sin(i * 0.071f + 0.3f)).ToArray();
        using ArcBuffer dyGpu = lane.Upload(dy);

        float[][] Run(bool persistent)
        {
            float[] Initial(int n) => Enumerable.Repeat(0.03125f, n).ToArray();
            using ArcBuffer dqkv = lane.Upload(Initial(data.Qkv.Length));
            using ArcBuffer dgate = lane.Upload(Initial(data.Gate.Length));
            using ArcBuffer dalpha = lane.Upload(Initial(data.Alpha.Length));
            using ArcBuffer dbeta = lane.Upload(Initial(data.Beta.Length));
            using var training = buffers.Forward(lane, data);
            training.UsePersistentBackward = persistent;
            long live = lane.AllocatedBytes, uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
            training.Backward(dyGpu, dqkv, dgate, dalpha, dbeta);
            training.Backward(dyGpu, dqkv, dgate, dalpha, dbeta);
            Assert.Equal(live, lane.AllocatedBytes);
            Assert.Equal(uploaded, lane.H2DBytes);
            Assert.Equal(downloaded, lane.D2HBytes);
            return [Read(lane, dqkv, data.Qkv.Length), Read(lane, dgate, data.Gate.Length),
                Read(lane, dalpha, data.Alpha.Length), Read(lane, dbeta, data.Beta.Length)];
        }

        float[][] reference = Run(false), optimized = Run(true);
        for (int i = 0; i < reference.Length; i++)
            AssertClose(reference[i], optimized[i], 3e-5f, 3e-5f);
    }

    private sealed class Data
    {
        internal readonly int Keys, Heads, Width, Kernel, Sequence, Channels, Values;
        internal readonly float[] Qkv, Gate, Alpha, Beta, Conv, Dt, A, Norm;
        internal readonly Qwen35GgufDescriptor Descriptor;

        internal Data(int keys, int heads, int width, int kernel, int sequence)
        {
            Keys = keys; Heads = heads; Width = width; Kernel = kernel; Sequence = sequence;
            Channels = (2 * keys + heads) * width; Values = heads * width;
            var random = new Random(19537 + width);
            float[] Random(int count, float scale) => Enumerable.Range(0, count)
                .Select(_ => (float)(2 * random.NextDouble() - 1) * scale).ToArray();
            Qkv = Random(sequence * Channels, 1.2f); Gate = Random(sequence * Values, 0.8f);
            Alpha = Random(sequence * heads, 0.8f); Beta = Random(sequence * heads, 0.8f);
            Conv = Random(Channels * kernel, 0.7f); Dt = Random(heads, 0.3f);
            A = Random(heads, 0.3f).Select(x => -MathF.Abs(x) - 0.3f).ToArray();
            Norm = Random(width, 0.3f).Select(x => x + 1f).ToArray();
            Descriptor = new(16, 1, 8, 2, 1, 4, 128, 16, 1e-4f, 10000f, 4,
                keys, heads, width, kernel, 4, []);
        }

        internal float[] Reference()
        {
            var history = new float[Channels * (Kernel - 1)];
            var state = new float[Values * Width];
            var output = new float[Sequence * Values];
            for (int t = 0; t < Sequence; ++t)
            {
                float[] token = Qwen35Math.DeltaStep(Qkv.AsSpan(t * Channels, Channels).ToArray(),
                    Gate.AsSpan(t * Values, Values).ToArray(), Alpha.AsSpan(t * Heads, Heads).ToArray(),
                    Beta.AsSpan(t * Heads, Heads).ToArray(), Conv, Dt, A, Norm,
                    history, state, Keys, Heads, Width, Kernel, Descriptor.RmsEpsilon);
                token.CopyTo(output, t * Values);
            }
            return output;
        }
    }

    private sealed class Inputs : IDisposable
    {
        internal readonly ArcBuffer Qkv, Gate, Alpha, Beta, Conv, Dt, A, Norm;
        internal Inputs(ArcExecutionLane lane, Data data)
        {
            Qkv = lane.Upload(data.Qkv); Gate = lane.Upload(data.Gate);
            Alpha = lane.Upload(data.Alpha); Beta = lane.Upload(data.Beta);
            Conv = lane.Upload(data.Conv); Dt = lane.Upload(data.Dt);
            A = lane.Upload(data.A); Norm = lane.Upload(data.Norm);
        }
        internal Qwen35TrainingDelta Forward(ArcExecutionLane lane, Data data, bool captureCheckpoints = true)
            => new(lane, Qkv, Gate, Alpha, Beta, Conv, Dt, A, Norm,
                data.Descriptor, data.Sequence, captureCheckpoints);
        public void Dispose()
        { Qkv.Dispose(); Gate.Dispose(); Alpha.Dispose(); Beta.Dispose(); Conv.Dispose(); Dt.Dispose(); A.Dispose(); Norm.Dispose(); }
    }

    private static double Loss(float[] values, float[] dy)
        => values.Select((v, i) => (double)v * dy[i]).Sum();

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var result = new float[count]; lane.Read(buffer, result); return result;
    }

    private static void AssertClose(float[] expected, float[] actual, float absolute, float relative)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(actual[i] - expected[i]) <= absolute + relative * MathF.Abs(expected[i]),
                $"Mismatch at {i}: expected {expected[i]:R}, actual {actual[i]:R}.");
    }
}
