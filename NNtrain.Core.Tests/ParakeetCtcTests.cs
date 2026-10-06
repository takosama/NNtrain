using System.Text.Json;
using NNtrain.Audio;
using Xunit;
namespace NNtrain.Core.Tests;
public sealed class ParakeetCtcTests
{
    [Fact]
    public void SubsamplingMasksBiasedInvalidRowsIndependentlyOfZeroPaddingExtent()
    {
        using var fixture = new Fixture();
        using var model = ParakeetCtcModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        float[][] valid = Enumerable.Range(0, 10).Select(t => Enumerable.Range(0, 80)
            .Select(b => .1f * MathF.Sin(t * .13f + b * .07f)).ToArray()).ToArray();
        float[][] shortPadding = valid.Concat(new[] { new float[80] }).ToArray();
        float[][] longPadding = valid.Concat(Enumerable.Range(0, 6).Select(_ => new float[80])).ToArray();
        float[][] expected = model.Encode(shortPadding, TestContext.Current.CancellationToken, validMelFrames: 10);
        float[][] actual = model.Encode(longPadding, TestContext.Current.CancellationToken, validMelFrames: 10);
        Assert.Equal(2, expected.Length);
        Assert.Equal(expected.Length, actual.Length);
        for (int t = 0; t < actual.Length; t++)
            for (int d = 0; d < actual[t].Length; d++) Assert.InRange(Math.Abs(actual[t][d] - expected[t][d]), 0, 1e-6);
    }

    [Fact]
    public void CtcCollapsesAdjacentRepeatsButKeepsRepeatsSeparatedByBlank()
    {
        using var fixture = new Fixture(); using var model = ParakeetCtcModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        Assert.Equal("ああ", model.DecodeCtc(new float[][] { [0, 10, 0], [0, 10, 0], [0, 0, 10], [0, 10, 0] }));
    }
    [Fact]
    public void AudioTraversesFrontendEncoderCtcAndFinalization()
    {
        using var fixture = new Fixture(); using var model = ParakeetCtcModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        float[] samples = Enumerable.Range(0, 16000).Select(i => .1f * MathF.Sin(i * .1f)).ToArray();
        Assert.Equal("あ", model.Transcribe(samples, TestContext.Current.CancellationToken));
        var stream = model.CreateStream(); Assert.Empty(stream.Append(samples, ct: TestContext.Current.CancellationToken));
        Assert.Equal("あ", stream.Append([], final: true, ct: TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => stream.Append([], ct: TestContext.Current.CancellationToken));
    }
    [Fact]
    public void BufferedPreviewCannotReuseCancelledOrOversizedUtterance()
    {
        using var fixture = new Fixture(); using var model = ParakeetCtcModel.Load(fixture.Directory, TestContext.Current.CancellationToken);
        var stream = model.CreateStream();
        Assert.Throws<OperationCanceledException>(() => stream.Append([], ct: new CancellationToken(true)));
        Assert.Throws<InvalidOperationException>(() => stream.Append([], ct: TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => model.CreateStream().Append(new float[16000 * 31], ct: TestContext.Current.CancellationToken));
    }
    [Fact]
    public void FrontendHasTerminalFrameAndFinitePerFeatureNormalization()
    {
        float[][] mel = ParakeetMel.Extract(Enumerable.Range(0, 16000).Select(i => .1f * MathF.Sin(i * .03f)).ToArray(), TestContext.Current.CancellationToken);
        Assert.Equal(101, mel.Length); Assert.All(mel, row => { Assert.Equal(80, row.Length); Assert.All(row, x => Assert.True(float.IsFinite(x))); });
        Assert.All(mel[^1], value => Assert.Equal(0, value));
        for (int b = 0; b < 80; b++) Assert.True(Math.Abs(mel[..^1].Average(row => (double)row[b])) < .0001);
    }
    internal sealed class Fixture : IDisposable
    {
        public string Directory { get; } = Path.Combine(Path.GetTempPath(), "nntrain-mini-parakeet-" + Guid.NewGuid().ToString("N"));
        public Fixture()
        {
            System.IO.Directory.CreateDirectory(Directory);
            File.WriteAllText(Path.Combine(Directory, "model_config.yaml"), "preprocessor:\n  sample_rate: 16000\n  features: 80\n  normalize: per_feature\n  window_size: 0.025\n  window_stride: 0.01\n  window: hann\n  n_fft: 512\nencoder:\n  d_model: 8\n  n_layers: 2\n  n_heads: 2\n  subsampling_conv_channels: 2\n  conv_kernel_size: 9\n  subsampling: dw_striding\n  subsampling_factor: 8\n  self_attention_model: rel_pos\n  conv_norm_type: batch_norm\n  xscaling: true\n");
            File.WriteAllText(Path.Combine(Directory, "tokenizer.vocab"), "▁\t0\nあ\t0\n");
            var tensors = new Dictionary<string, (int[] Shape, Half[] Values)>(); var random = new Random(723);
            void Add(string name, params int[] shape)
            { int count = shape.Aggregate(1, (x, y) => x * y); tensors.Add(name, (shape, Enumerable.Range(0, count).Select(_ => (Half)((random.NextDouble() - .5) * .02)).ToArray())); }
            void Matrix(string name, int output, int input, bool pointwise = false)
            { Add(name + ".weight", pointwise ? [output, input, 1] : [output, input]); Add(name + ".bias", output); }
            void Norm(string name)
            { Add(name + ".weight", 8); Array.Fill(tensors[name + ".weight"].Values, (Half)1); Add(name + ".bias", 8); Array.Clear(tensors[name + ".bias"].Values); }
            Add("encoder.pre_encode.conv.0.weight", 2, 1, 3, 3); Add("encoder.pre_encode.conv.0.bias", 2);
            foreach (int stage in new[] { 2, 5 })
            { Add($"encoder.pre_encode.conv.{stage}.weight", 2, 1, 3, 3); Add($"encoder.pre_encode.conv.{stage}.bias", 2); Add($"encoder.pre_encode.conv.{stage + 1}.weight", 2, 2, 1, 1); Add($"encoder.pre_encode.conv.{stage + 1}.bias", 2); }
            Matrix("encoder.pre_encode.out", 8, 20);
            for (int layer = 0; layer < 2; layer++)
            {
                string prefix = $"encoder.layers.{layer}";
                foreach (string norm in new[] { "norm_feed_forward1", "norm_feed_forward2", "norm_self_att", "norm_conv", "norm_out" }) Norm(prefix + "." + norm);
                foreach (int ff in new[] { 1, 2 }) { Matrix(prefix + $".feed_forward{ff}.linear1", 32, 8); Matrix(prefix + $".feed_forward{ff}.linear2", 8, 32); }
                foreach (string projection in new[] { "linear_q", "linear_k", "linear_v", "linear_out" }) Matrix(prefix + ".self_attn." + projection, 8, 8);
                Add(prefix + ".self_attn.linear_pos.weight", 8, 8); Add(prefix + ".self_attn.pos_bias_u", 2, 4); Add(prefix + ".self_attn.pos_bias_v", 2, 4);
                Matrix(prefix + ".conv.pointwise_conv1", 16, 8, true); Matrix(prefix + ".conv.pointwise_conv2", 8, 8, true);
                Add(prefix + ".conv.depthwise_conv.weight", 8, 1, 9); Add(prefix + ".conv.depthwise_conv.bias", 8);
                Norm(prefix + ".conv.batch_norm"); Add(prefix + ".conv.batch_norm.running_mean", 8); Array.Clear(tensors[prefix + ".conv.batch_norm.running_mean"].Values);
                Add(prefix + ".conv.batch_norm.running_var", 8); Array.Fill(tensors[prefix + ".conv.batch_norm.running_var"].Values, (Half)1);
            }
            Matrix("ctc_decoder.decoder_layers.0", 3, 8, true);
            tensors["ctc_decoder.decoder_layers.0.bias"].Values[1] = (Half)20;
            var header = new Dictionary<string, object>(); long offset = 0;
            foreach (var (name, tensor) in tensors) { header.Add(name, new { dtype = "F16", shape = tensor.Shape, data_offsets = new[] { offset, offset + tensor.Values.Length * 2L } }); offset += tensor.Values.Length * 2L; }
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            using var file = File.Create(Path.Combine(Directory, "model.safetensors")); using var writer = new BinaryWriter(file);
            writer.Write((ulong)json.Length); writer.Write(json);
            foreach (var tensor in tensors.Values) foreach (Half value in tensor.Values) writer.Write(BitConverter.HalfToUInt16Bits(value));
        }
        public void Dispose()
        { foreach (string name in new[] { "model_config.yaml", "tokenizer.vocab", "model.safetensors" }) File.Delete(Path.Combine(Directory, name)); System.IO.Directory.Delete(Directory); }
    }
}
