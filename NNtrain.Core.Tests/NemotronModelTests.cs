using System.Text.Json;
using NNtrain.Audio;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class NemotronModelTests
{
    [Fact]
    public void ShortFinalChunkConsumesAllPaddedEncoderFrames()
    {
        using var fixture = new MiniAsrFixture();
        using var model = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        var stream = model.CreateStream();
        stream.Append(new float[1600], final: true, ct: TestContext.Current.CancellationToken);
        Assert.Equal(4, stream.DecodedEncoderFrames);
        Assert.Equal(stream.EncoderFrames, stream.DecodedEncoderFrames);
    }
    [Fact]
    public void ConformerSubsamplingAndCachesMatchWholeUtterance()
    {
        using var fixture = new MiniAsrFixture();
        using var model = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        float[][] mel = Enumerable.Range(0, 57).Select(t => Enumerable.Range(0, 128).Select(f => MathF.Sin(t * .1f + f * .03f)).ToArray()).ToArray();
        float[][] expected = model.Encode(mel, model.CreateStream(), TestContext.Current.CancellationToken);
        var streaming = model.CreateStream();
        float[][] actual = model.Encode(mel[..25], streaming, TestContext.Current.CancellationToken)
            .Concat(model.Encode(mel[25..], streaming, TestContext.Current.CancellationToken)).ToArray();
        Assert.Equal(8, actual.Length);
        for (int t = 0; t < actual.Length; t++)
            for (int d = 0; d < actual[t].Length; d++) Assert.InRange(Math.Abs(actual[t][d] - expected[t][d]), 0, 2e-5);
    }

    [Fact]
    public void BlankDoesNotAdvanceLstmAndFinalizationCannotBeReused()
    {
        using var fixture = new MiniAsrFixture();
        using var model = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        var stream = model.CreateStream();
        string text = stream.Append(new float[16000], true, ct: TestContext.Current.CancellationToken);
        Assert.Equal("", text);
        Assert.Equal(1, stream.DecoderEvaluations);
        Assert.Throws<InvalidOperationException>(() => stream.Append([], ct: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CancellationClosesMutatedUtterance()
    {
        using var fixture = new MiniAsrFixture();
        using var model = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        var stream = model.CreateStream();
        Assert.Throws<OperationCanceledException>(() => stream.Append(new float[5000], ct: new CancellationToken(true)));
        Assert.Throws<InvalidOperationException>(() => stream.Append([], ct: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CacheStorageStopsGrowingAfterSlidingWindowFills()
    {
        using var fixture = new MiniAsrFixture();
        using var model = NemotronAsrModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        var stream = model.CreateStream();
        model.Encode(new float[25].Select(_ => new float[128]).ToArray(), stream, TestContext.Current.CancellationToken);
        long previous = 0;
        for (int chunk = 0; chunk < 25; chunk++)
        {
            model.Encode(new float[32].Select(_ => new float[128]).ToArray(), stream, TestContext.Current.CancellationToken);
            if (chunk >= 15) Assert.Equal(previous, stream.CacheBytes);
            previous = stream.CacheBytes;
        }
    }

    internal sealed class MiniAsrFixture : IDisposable
    {
        public string Directory { get; } = Path.Combine(Path.GetTempPath(), "nntrain-mini-asr-" + Guid.NewGuid().ToString("N"));
        public MiniAsrFixture()
        {
            System.IO.Directory.CreateDirectory(Directory);
            var config = new { model_type = "nemotron3_5_asr", decoder_hidden_size = 4, num_decoder_layers = 2,
                vocab_size = 3, blank_token_id = 2, num_prompts = 16, max_symbols_per_step = 10,
                encoder_config = new { hidden_size = 8, num_hidden_layers = 2, num_attention_heads = 2,
                    num_key_value_heads = 2, subsampling_conv_channels = 2, conv_kernel_size = 9,
                    sliding_window = 57, subsampling_factor = 8, subsampling_conv_kernel_size = 3,
                    subsampling_conv_stride = 2, num_mel_bins = 128 } };
            File.WriteAllText(Path.Combine(Directory, "config.json"), JsonSerializer.Serialize(config));
            File.WriteAllText(Path.Combine(Directory, "tokenizer.json"), JsonSerializer.Serialize(new {
                model = new { type = "BPE", vocab = new Dictionary<string, int> { ["▁"] = 0, ["こんにちは"] = 1, ["<blank>"] = 2 } },
                added_tokens = new[] { new { id = 2, content = "<blank>", special = true } } }));
            var weights = new Dictionary<string, (int[] Shape, Half[] Values)>();
            var random = new Random(6006);
            void Add(string name, params int[] shape)
            {
                int count = shape.Aggregate(1, (a, b) => a * b);
                bool norm = name.Contains("norm", StringComparison.Ordinal) && name.EndsWith(".weight", StringComparison.Ordinal);
                weights[name] = (shape, Enumerable.Range(0, count).Select(_ => norm ? (Half)1 : (Half)((random.NextSingle() - .5f) * .1f)).ToArray());
            }
            void Linear(string name, int outputs, int inputs, bool bias = false)
            { Add(name + ".weight", outputs, inputs); if (bias) Add(name + ".bias", outputs); }
            void Norm(string name) { Add(name + ".weight", 8); Add(name + ".bias", 8); }
            Add("encoder.subsampling.conv_in.weight", 2, 1, 3, 3); Add("encoder.subsampling.conv_in.bias", 2);
            for (int i = 0; i < 2; i++)
            {
                Add($"encoder.subsampling.layers.{i}.depthwise_conv.weight", 2, 1, 3, 3);
                Add($"encoder.subsampling.layers.{i}.depthwise_conv.bias", 2);
                Add($"encoder.subsampling.layers.{i}.pointwise_conv.weight", 2, 2, 1, 1);
                Add($"encoder.subsampling.layers.{i}.pointwise_conv.bias", 2);
            }
            Linear("encoder.subsampling.linear", 8, 34, true);
            for (int layer = 0; layer < 2; layer++)
            {
                string p = $"encoder.layers.{layer}";
                foreach (string ff in new[] { "feed_forward1", "feed_forward2" })
                { Linear(p + "." + ff + ".linear1", 32, 8); Linear(p + "." + ff + ".linear2", 8, 32); }
                foreach (string norm in new[] { "norm_feed_forward1", "norm_feed_forward2", "norm_self_att", "norm_conv", "norm_out", "conv.norm" }) Norm(p + "." + norm);
                foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj" }) Linear(p + ".self_attn." + proj, 8, 8);
                Add(p + ".self_attn.bias_u", 2, 4); Add(p + ".self_attn.bias_v", 2, 4);
                Add(p + ".conv.pointwise_conv1.weight", 16, 8, 1);
                Add(p + ".conv.depthwise_conv.weight", 8, 1, 9);
                Add(p + ".conv.pointwise_conv2.weight", 8, 8, 1);
            }
            Linear("prompt_projector.linear_1", 16, 24, true); Linear("prompt_projector.linear_2", 8, 16, true);
            Linear("encoder_projector", 4, 8, true); Add("decoder.embedding.weight", 3, 4);
            for (int layer = 0; layer < 2; layer++)
            {
                Add($"decoder.lstm.weight_ih_l{layer}", 16, 4); Add($"decoder.lstm.weight_hh_l{layer}", 16, 4);
                Add($"decoder.lstm.bias_ih_l{layer}", 16); Add($"decoder.lstm.bias_hh_l{layer}", 16);
            }
            Linear("decoder.decoder_projector", 4, 4, true); Linear("joint.head", 3, 4, true);
            weights["joint.head.weight"] = ([3, 4], new Half[12]);
            weights["joint.head.bias"] = ([3], [(Half)(-10), (Half)(-10), (Half)10]);
            long offset = 0;
            var header = new Dictionary<string, object>();
            foreach (var (name, weight) in weights)
            {
                long end = offset + weight.Values.Length * 2;
                header[name] = new { dtype = "F16", shape = weight.Shape, data_offsets = new[] { offset, end } };
                offset = end;
            }
            byte[] headerBytes = JsonSerializer.SerializeToUtf8Bytes(header);
            using var writer = new BinaryWriter(File.Create(Path.Combine(Directory, "model.safetensors")));
            writer.Write((ulong)headerBytes.Length); writer.Write(headerBytes);
            foreach (var weight in weights.Values) foreach (Half value in weight.Values) writer.Write(BitConverter.HalfToUInt16Bits(value));
        }
        public void Dispose() => System.IO.Directory.Delete(Directory, true);
    }
}
