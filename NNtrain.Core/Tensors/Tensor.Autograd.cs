namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Runs reverse-mode differentiation from this tensor.
    /// </summary>
    /// <remarks>
    /// Reachable intermediate gradients are cleared for each traversal. Leaf
    /// gradients accumulate until explicitly cleared with <see cref="ZeroGrad"/>.
    /// This graph may be traversed repeatedly while its Tensor data remains
    /// unchanged. A Parameter update requires a new forward graph.
    /// </remarks>
    /// <param name="seed">
    /// The output gradient. It is required when this tensor has more than one
    /// element and must contain exactly <see cref="Numel"/> values.
    /// </param>
    public void Backward(float[]? seed = null)
        => AutogradEngine.Backward(this, seed);

    public void backward(float[]? gradient = null) => Backward(gradient);
}
