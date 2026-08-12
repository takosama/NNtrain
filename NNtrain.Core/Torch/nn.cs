#pragma warning disable CS8981

namespace NNtrain;

/// <summary>PyTorch-style neural-network factories.</summary>
public static class nn
{
    public static class functional
    {
        public static Tensor cross_entropy(
            Tensor input,
            int[] target,
            float label_smoothing = 0f,
            int ignore_index = Tensor.DefaultCrossEntropyIgnoreIndex)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(target);
            return input.CrossEntropyWithLogits(
                target,
                label_smoothing,
                ignore_index);
        }
    }

    public static TransformerClassifier transformer_classifier(
        int seq_len,
        int d_model,
        int num_heads,
        int dim_feedforward,
        int num_layers,
        int num_classes,
        float dropout = 0f,
        float init_scale = 0.02f,
        Random? generator = null)
        => new(
            seqLen: seq_len,
            dModel: d_model,
            numHeads: num_heads,
            dHidden: dim_feedforward,
            numLayers: num_layers,
            numClasses: num_classes,
            rng: generator ?? torch.generator(),
            initScale: init_scale,
            dropout: dropout);

    public static Dropout dropout(
        float p = 0.5f,
        Random? generator = null)
        => new(p, generator ?? torch.generator());

    public static GptRinWikiJp transformer_lm(
        int vocab_size,
        int context_length,
        int d_model,
        int num_heads,
        int dim_feedforward,
        int num_layers,
        float dropout = 0f,
        float init_scale = 0.02f,
        Random? generator = null)
        => new(
            vocab_size,
            context_length,
            d_model,
            num_heads,
            dim_feedforward,
            num_layers,
            generator ?? torch.generator(),
            init_scale,
            dropout);

    public static HyenaGpt hyena_lm(
        int vocab_size,
        int context_length,
        int d_model,
        int dim_feedforward,
        int num_layers,
        int filter_width = 64,
        HyenaConvolutionAlgorithm convolution =
            HyenaConvolutionAlgorithm.Auto,
        float dropout = 0f,
        float init_scale = 0.02f,
        Random? generator = null)
        => new(
            vocab_size,
            context_length,
            d_model,
            dim_feedforward,
            num_layers,
            generator ?? torch.generator(),
            init_scale,
            dropout,
            filter_width,
            convolution);

    public static ForgetScanGpt forget_scan_lm(
        int vocab_size,
        int context_length,
        int d_model,
        int dim_feedforward,
        int num_layers,
        float dropout = 0f,
        float init_scale = 0.02f,
        Random? generator = null)
        => new(
            vocab_size,
            context_length,
            d_model,
            dim_feedforward,
            num_layers,
            generator ?? torch.generator(),
            init_scale,
            dropout);

    public static FrogetMemoryV2Gpt forget_memory_v2_lm(
        int vocab_size,
        int context_length,
        int d_model,
        int dim_feedforward,
        int num_layers,
        int key_width = 16,
        int value_width = 16,
        float retention_min = 0.5f,
        float retention_max = 0.99f,
        float dropout = 0f,
        float init_scale = 0.02f,
        Random? generator = null)
        => new(
            vocab_size,
            context_length,
            d_model,
            dim_feedforward,
            num_layers,
            key_width,
            value_width,
            retention_min,
            retention_max,
            generator ?? torch.generator(),
            init_scale,
            dropout);
}
