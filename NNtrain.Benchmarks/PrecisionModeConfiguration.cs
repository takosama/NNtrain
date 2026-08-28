using System.Text.Json;

namespace NNtrain.Benchmarks;

internal static class PrecisionModeConfiguration
{
    internal static TensorPrecisionMode Read(JsonElement root)
    {
        if (root.TryGetProperty("precisionMode", out JsonElement precision))
        {
            if (root.TryGetProperty("modelDType", out _))
            {
                throw new InvalidDataException(
                    "precisionMode cannot be combined with legacy modelDType.");
            }
            string value = precision.GetString()
                ?? throw new InvalidDataException(
                    "precisionMode must be a string.");
            try
            {
                return TensorPrecisionModeNames.Parse(value);
            }
            catch (ArgumentException exception)
            {
                throw new InvalidDataException(exception.Message, exception);
            }
        }

        if (!root.TryGetProperty("modelDType", out JsonElement legacy)
            || legacy.ValueKind == JsonValueKind.Null)
        {
            return TensorPrecisionMode.Mix16_32;
        }
        string legacyValue = legacy.GetString()
            ?? throw new InvalidDataException("modelDType must be a string.");
        return legacyValue.ToLowerInvariant() switch
        {
            "float16" or "half" => TensorPrecisionMode.Mix16_32,
            "bfloat16" or "bf16" => TensorPrecisionMode.BFloat16,
            "float32" => TensorPrecisionMode.Float32,
            _ => throw new InvalidDataException(
                $"Unsupported legacy modelDType '{legacyValue}'."),
        };
    }
}
