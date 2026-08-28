using System.Text.Json;

namespace NNtrain;

/// <summary>
/// Writes a module state directly from parameter storage through one bounded
/// staging allocation.  The resulting JSON is byte-compatible with
/// <see cref="ModuleState"/>, but no model-sized host object graph is built.
/// </summary>
internal static class StreamingModuleStateJsonWriter
{
    internal static void Write(
        Utf8JsonWriter writer,
        Module model,
        CheckpointFloatStagingBuffer staging)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(staging);

        Parameter[] parameters = model.parameters().ToArray();
        writer.WriteStartObject();
        WriteProperty(
            writer,
            nameof(ModuleState.FormatVersion),
            ModuleState.CurrentFormatVersion);
        writer.WritePropertyName(nameof(ModuleState.Parameters));
        writer.WriteStartArray();
        for (int index = 0; index < parameters.Length; index++)
        {
            Parameter parameter = parameters[index];
            Tensor tensor = parameter.T;
            writer.WriteStartObject();
            WriteProperty(writer, nameof(ModuleParameterState.Index), index);
            WriteProperty(
                writer,
                nameof(ModuleParameterState.Name),
                parameter.Name);
            WriteProperty(
                writer,
                nameof(ModuleParameterState.Shape),
                tensor.Shape.ToArray());
            writer.WritePropertyName(nameof(ModuleParameterState.Values));
            writer.WriteStartArray();
            int maximumChunkElements = tensor.DType == TensorDType.Bfp8
                ? CheckpointFloatStagingBuffer.MaximumElementCount / 2
                : CheckpointFloatStagingBuffer.MaximumElementCount;
            for (int offset = 0; offset < tensor.Numel;)
            {
                int count = Math.Min(
                    maximumChunkElements,
                    tensor.Numel - offset);
                ReadOnlySpan<float> values = tensor.CopyCheckpointRangeTo(
                    offset,
                    count,
                    staging,
                    preferMaster: true);
                foreach (float value in values)
                    torch.SerializeJsonValue(writer, value);
                offset += count;
            }
            writer.WriteEndArray();
            WriteProperty(
                writer,
                nameof(ModuleParameterState.DType),
                tensor.DType);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    internal static void WriteProperty<T>(
        Utf8JsonWriter writer,
        string name,
        T value)
    {
        writer.WritePropertyName(name);
        torch.SerializeJsonValue(writer, value);
    }
}
