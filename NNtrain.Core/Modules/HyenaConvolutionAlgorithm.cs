namespace NNtrain;

/// <summary>
/// Selects the long-convolution implementation used by Hyena.
/// </summary>
public enum HyenaConvolutionAlgorithm
{
    /// <summary>Chooses the faster implementation from the sequence length.</summary>
    Auto,

    /// <summary>Uses the quadratic direct causal convolution.</summary>
    Direct,

    /// <summary>Uses a zero-padded SIMD FFT convolution.</summary>
    Fft,
}
