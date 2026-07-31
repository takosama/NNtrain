# Tensor semantics

`Tensor` stores dense `float` values in row-major order and builds a reverse-mode
automatic-differentiation graph for every supported operation.

Graph ownership and execution responsibilities are defined in
[autograd-design.md](autograd-design.md).

## Shape

- A shape contains at least one dimension.
- Every dimension is a positive `Int32` value.
- The product of all dimensions must fit in `Int32`.
- The product of the dimensions must equal the data length.
- A scalar is represented as shape `[1]`; rank-zero tensors are not supported.
- Shape arrays passed to a constructor are copied.

## Data and gradients

- Data passed to the public constructor is copied.
- `Data`, `Grad`, and `Shape` expose read-only views.
- Tensor operations and optimizers mutate the underlying buffers only through
  internal APIs.
- Calling `Backward` clears reachable non-leaf gradients before computing the
  new gradients.
- Leaf gradients accumulate until explicitly cleared with `ZeroGrad`.
- A graph can run `Backward` repeatedly while its Tensor data is unchanged.
- Updating a Parameter invalidates graphs built from its previous value; callers
  must perform a fresh forward pass before the next `Backward`.
- Operations inside `AutogradContext.NoGrad()` compute normal forward values but
  return detached leaves that do not propagate gradients to their inputs.
- A non-scalar output requires a seed containing one value per output element.

## Broadcasting

Element-wise arithmetic supports only:

1. operands with exactly equal shapes; or
2. operands where either side contains exactly one element.

A one-element tensor is scalar-like regardless of its rank. General NumPy-style
dimension broadcasting is not supported.

## Supported ranks

- `Slice`, `Concat`, softmax, log-softmax, and layer normalization support rank
  one and rank two as documented by their APIs.
- `Transpose` requires rank two.
- `MatMul` supports rank1 × rank1, rank2 × rank1, and rank2 × rank2.
- Unsupported rank combinations throw `NotSupportedException`.
- Invalid argument values throw `ArgumentException` or
  `ArgumentOutOfRangeException`; an operation applied to the wrong tensor rank
  throws `InvalidOperationException`.
