namespace NNtrain;

/// <summary>
/// Describes the physical representation used by a tensor storage.
/// </summary>
/// <remarks>
/// Float8 and smaller formats are reserved here so storage, checkpoint, and
/// kernel dispatch contracts do not need to be redesigned after Float16.
/// Their codecs are intentionally not enabled until their quantization and
/// scaling policies are implemented and tested.
/// </remarks>
public enum TensorDType
{
    Float32 = 0,
    Float16 = 1,
    Float8E4M3Fn = 2,
    Float8E5M2 = 3,
    Float4 = 4,
    Float2 = 5,
    Ternary1Bit58 = 6,
}

internal static class TensorDTypeContract
{
    internal static bool IsImplemented(TensorDType dtype)
        => dtype is TensorDType.Float32 or TensorDType.Float16;

    internal static void ValidateImplemented(
        TensorDType dtype,
        string parameterName)
    {
        if (!Enum.IsDefined(dtype))
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                dtype,
                "Unknown tensor dtype.");
        }

        if (!IsImplemented(dtype))
        {
            throw new NotSupportedException(
                $"Tensor dtype '{dtype}' is reserved for a future storage " +
                "codec and is not implemented yet.");
        }
    }

    internal static TensorDType Promote(
        IReadOnlyList<Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        TensorDType result = TensorDType.Float32;
        if (tensors.Count == 0)
            return result;

        result = TensorDType.Float16;
        for (int index = 0; index < tensors.Count; index++)
        {
            Tensor tensor = tensors[index]
                ?? throw new ArgumentException(
                    "Tensor list cannot contain null.",
                    nameof(tensors));
            if (tensor.DType == TensorDType.Float32)
                return TensorDType.Float32;
            if (tensor.DType != TensorDType.Float16)
            {
                throw new NotSupportedException(
                    $"Tensor dtype '{tensor.DType}' does not have a " +
                    "promotion rule yet.");
            }
            result = tensor.DType;
        }
        return result;
    }
}
