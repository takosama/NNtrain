using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;
using NNtrain;
using Xunit;

public sealed class OptimizerStateStreamTests
{
    [Fact]
    public void BinaryRoundTripPreservesEverySupportedOptimizerExactly()
    {
        foreach ((IOptimizer source, IOptimizer restored) in CreatePairs())
        {
            source.step();
            using var stream = new MemoryStream();

            OptimizerStateStream.SaveStateBinary(source, stream);
            stream.Position = 0;
            OptimizerStateStream.LoadStateBinary(restored, stream);

            Assert.Equal(
                source.state_dict().StateJsonText,
                restored.state_dict().StateJsonText);
            Assert.Equal(stream.Length, stream.Position);
        }
    }

    [Fact]
    public void BinaryStateIsSmallerThanFloatArrayJson()
    {
        Parameter parameter = CreateParameter("matrix", [32, 32]);
        for (int index = 0; index < parameter.T.Numel; index++)
            parameter.T.MutableGrad[index] = (index + 0.25f) / 97f;
        var optimizer = new NekoMuon(
            [parameter],
            new NekoMuonOptions
            {
                NewtonSchulzInterval = 100,
                WeightDecay = 0f,
            });
        optimizer.step();
        using var binary = new MemoryStream();
        using var json = new MemoryStream();

        OptimizerStateStream.SaveStateBinary(optimizer, binary);
        OptimizerStateStream.SaveStateJson(optimizer, json);

        Assert.True(binary.Length < json.Length * 0.6);
    }

    [Fact]
    public void BFloat16AdamWBinarySaveKeepsV1AndWritesFixedSizeChunks()
    {
        int length = OptimizerStateStream.BFloat16ConversionChunkElements * 3
            + 17;
        Parameter parameter = CreateParameter("large", [length]);
        for (int index = 0; index < length; index++)
            parameter.T.MutableGrad[index] = (index % 31 - 15) / 31f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                WeightDecay = 0f,
                UseBFloat16FirstMoment = true,
                UseBFloat16SecondMoment = true,
            });
        optimizer.step();
        using var stream = new MaximumWriteRecordingStream();

        OptimizerStateStream.SaveStateBinary(optimizer, stream);

        byte[] payload = stream.ToArray();
        Assert.Equal(1, BitConverter.ToInt32(payload, 8));
        Assert.True(
            stream.MaximumWriteLength
                <= OptimizerStateStream.BFloat16ConversionChunkElements
                    * sizeof(float),
            $"Largest write was {stream.MaximumWriteLength:N0} bytes.");
    }

    [Fact]
    public void BinaryLoaderRejectsWrongOptimizerTypeAndTruncation()
    {
        var source = new Lion([CreateParameter("weight", [2])]);
        using var stream = new MemoryStream();
        OptimizerStateStream.SaveStateBinary(source, stream);
        byte[] payload = stream.ToArray();

        var wrong = new AdamW([CreateParameter("weight", [2])]);
        Assert.Throws<InvalidDataException>(() =>
            OptimizerStateStream.LoadStateBinary(
                wrong,
                new MemoryStream(payload)));

        var restored = new Lion([CreateParameter("weight", [2])]);
        Assert.ThrowsAny<Exception>(() =>
            OptimizerStateStream.LoadStateBinary(
                restored,
                new MemoryStream(payload[..^1])));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NekoMuonDepthPolicyRoundTripsThroughStreams(bool binary)
    {
        var source = new NekoMuon(
            [CreateParameter("neko", [1, 2])],
            new NekoMuonOptions
            {
                MaxNewtonSchulzSteps = 4,
            });
        source.SetNewtonSchulzDepthPolicy(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            1.5f);
        using var stream = new MemoryStream();

        if (binary)
            OptimizerStateStream.SaveStateBinary(source, stream);
        else
            OptimizerStateStream.SaveStateJson(source, stream);
        stream.Position = 0;
        var restored = new NekoMuon(
            [CreateParameter("neko", [1, 2])]);
        if (binary)
            OptimizerStateStream.LoadStateBinary(restored, stream);
        else
            OptimizerStateStream.LoadStateJson(restored, stream);

        NekoMuonOptions options = restored.CaptureState().Options;
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Fixed,
            options.NewtonSchulzDepthMode);
        Assert.Equal(1.5f, options.NewtonSchulzDepth);
        Assert.Equal(4, options.MaxNewtonSchulzSteps);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LegacyNekoMuonOptionsWithoutDepthPolicyUseAdaptiveDefaults(
        bool binary)
    {
        var source = new NekoMuon(
            [CreateParameter("neko", [1, 2])],
            new NekoMuonOptions
            {
                MaxNewtonSchulzSteps = 4,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 1.5f,
            });
        using var current = new MemoryStream();
        if (binary)
            OptimizerStateStream.SaveStateBinary(source, current);
        else
            OptimizerStateStream.SaveStateJson(source, current);
        byte[] legacy = binary
            ? RemoveDepthPolicyFromBinary(current.ToArray())
            : RemoveDepthPolicyFromJson(current.ToArray());
        var restored = new NekoMuon(
            [CreateParameter("neko", [1, 2])]);

        using var input = new MemoryStream(legacy);
        if (binary)
            OptimizerStateStream.LoadStateBinary(restored, input);
        else
            OptimizerStateStream.LoadStateJson(restored, input);

        NekoMuonOptions options = restored.CaptureState().Options;
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Adaptive,
            options.NewtonSchulzDepthMode);
        Assert.Equal(0f, options.NewtonSchulzDepth);
        Assert.Equal(4, options.MaxNewtonSchulzSteps);
    }

    private static IEnumerable<(IOptimizer Source, IOptimizer Restored)>
        CreatePairs()
    {
        Parameter adamSource = CreateParameter("adam", [2]);
        new float[] { 0.25f, -0.5f }
            .AsSpan()
            .CopyTo(adamSource.T.MutableGrad);
        yield return (
            new AdamW(
                [adamSource],
                new AdamWOptions
                {
                    LearningRate = 0.02f,
                    WeightDecay = 0f,
                }),
            new AdamW([CreateParameter("adam", [2])]));

        Parameter adamBFloat16Source = CreateParameter("adam-bf16", [3]);
        new float[] { 0.125f, -0.375f, 0.625f }
            .AsSpan()
            .CopyTo(adamBFloat16Source.T.MutableGrad);
        yield return (
            new AdamW(
                [adamBFloat16Source],
                new AdamWOptions
                {
                    LearningRate = 0.015f,
                    WeightDecay = 0f,
                    UseBFloat16FirstMoment = true,
                    UseBFloat16SecondMoment = true,
                }),
            new AdamW([CreateParameter("adam-bf16", [3])]));

        Parameter nekoSource = CreateParameter("neko", [1, 2]);
        new float[] { 0.4f, -0.2f }
            .AsSpan()
            .CopyTo(nekoSource.T.MutableGrad);
        yield return (
            new NekoMuon(
                [nekoSource],
                new NekoMuonOptions
                {
                    LearningRate = 0.03f,
                    NewtonSchulzInterval = 3,
                    WeightDecay = 0f,
                }),
            new NekoMuon([CreateParameter("neko", [1, 2])]));

        Parameter lionSource = CreateParameter("lion", [2]);
        new float[] { -0.3f, 0.7f }
            .AsSpan()
            .CopyTo(lionSource.T.MutableGrad);
        yield return (
            new Lion(
                [lionSource],
                new LionOptions
                {
                    LearningRate = 0.04f,
                    WeightDecay = 0f,
                }),
            new Lion([CreateParameter("lion", [2])]));

        Parameter gainSource = CreateParameter("gain", [2]);
        new float[] { 0.6f, -0.1f }
            .AsSpan()
            .CopyTo(gainSource.T.MutableGrad);
        yield return (
            new GainShareAdamW(
                [[gainSource]],
                new GainShareAdamWOptions
                {
                    LearningRate = 0.05f,
                    WeightDecay = 0f,
                }),
            new GainShareAdamW(
                [[CreateParameter("gain", [2])]]));
    }

    private static Parameter CreateParameter(string name, int[] shape)
        => new(
            new float[shape.Aggregate(1, (product, value) =>
                checked(product * value))],
            shape,
            name,
            WeightDecayPolicy.Exclude);

    private static byte[] RemoveDepthPolicyFromJson(byte[] payload)
    {
        JsonObject root = JsonNode.Parse(
            Encoding.UTF8.GetString(payload))!.AsObject();
        RemoveDepthPolicy(root["Options"]!.AsObject());
        return Encoding.UTF8.GetBytes(root.ToJsonString());
    }

    private static byte[] RemoveDepthPolicyFromBinary(byte[] payload)
    {
        const int MagicLength = 8;
        int typeLength = BinaryPrimitives.ReadInt32LittleEndian(
            payload.AsSpan(MagicLength + sizeof(int), sizeof(int)));
        int metadataLengthOffset = checked(
            MagicLength
                + sizeof(int)
                + sizeof(int)
                + typeLength
                + sizeof(int)
                + sizeof(int));
        int metadataLength = BinaryPrimitives.ReadInt32LittleEndian(
            payload.AsSpan(metadataLengthOffset, sizeof(int)));
        int metadataOffset = metadataLengthOffset + sizeof(int);
        JsonObject options = JsonNode.Parse(
            Encoding.UTF8.GetString(
                payload,
                metadataOffset,
                metadataLength))!.AsObject();
        RemoveDepthPolicy(options);
        byte[] legacyMetadata = Encoding.UTF8.GetBytes(
            options.ToJsonString());
        int suffixOffset = checked(metadataOffset + metadataLength);
        var legacy = new byte[
            payload.Length - metadataLength + legacyMetadata.Length];
        payload.AsSpan(0, metadataLengthOffset).CopyTo(legacy);
        BinaryPrimitives.WriteInt32LittleEndian(
            legacy.AsSpan(metadataLengthOffset, sizeof(int)),
            legacyMetadata.Length);
        legacyMetadata.CopyTo(
            legacy.AsSpan(metadataOffset, legacyMetadata.Length));
        payload.AsSpan(suffixOffset).CopyTo(
            legacy.AsSpan(metadataOffset + legacyMetadata.Length));
        return legacy;
    }

    private static void RemoveDepthPolicy(JsonObject options)
    {
        Assert.True(options.Remove("NewtonSchulzDepthMode"));
        Assert.True(options.Remove("NewtonSchulzDepth"));
    }

    private sealed class MaximumWriteRecordingStream : Stream
    {
        private readonly MemoryStream _inner = new();

        internal int MaximumWriteLength { get; private set; }

        internal byte[] ToArray() => _inner.ToArray();

        public override bool CanRead => false;
        public override bool CanSeek => true;
        public override bool CanWrite => true;
        public override long Length => _inner.Length;
        public override long Position
        {
            get => _inner.Position;
            set => _inner.Position = value;
        }

        public override void Flush() => _inner.Flush();

        public override int Read(byte[] buffer, int offset, int count)
            => throw new NotSupportedException();

        public override long Seek(long offset, SeekOrigin origin)
            => _inner.Seek(offset, origin);

        public override void SetLength(long value) => _inner.SetLength(value);

        public override void Write(byte[] buffer, int offset, int count)
        {
            MaximumWriteLength = Math.Max(MaximumWriteLength, count);
            _inner.Write(buffer, offset, count);
        }

        public override void Write(ReadOnlySpan<byte> buffer)
        {
            MaximumWriteLength = Math.Max(MaximumWriteLength, buffer.Length);
            _inner.Write(buffer);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
                _inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
