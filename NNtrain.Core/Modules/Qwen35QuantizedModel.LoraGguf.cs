using System.Runtime.InteropServices;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    /// <summary>Loads the F32 GGUF LoRA produced by export_qwen35_lora.py for inference.</summary>
    private void LoadLoraGguf(string path)
    {
        if (_options.LoraTraining)
            throw new NotSupportedException("A GGUF LoRA contains A/B weights only. Resume training with adapter.bin.");
        using var reader = new GgufReader(path);
        IReadOnlyDictionary<string, object> metadata = reader.Metadata;
        if (reader.Version != 3 ||
            !HasString("general.type", "adapter") ||
            !HasString("general.architecture", "qwen35") ||
            !HasString("adapter.type", "lora") ||
            !metadata.TryGetValue("general.file_type", out object? fileType) || fileType is not uint || (uint)fileType != 0 ||
            !metadata.TryGetValue("general.alignment", out object? alignment) || alignment is not uint || (uint)alignment != 32 ||
            !metadata.TryGetValue("adapter.lora.alpha", out object? alphaValue) || alphaValue is not float alpha ||
            !float.IsFinite(alpha) || alpha <= 0 ||
            !metadata.TryGetValue("nntrain.base_model.sha256", out object? baseHashValue) ||
            baseHashValue is not string baseHash || baseHash.Length != 64)
            throw new InvalidDataException("This is not a supported NNtrain Qwen3.5 F32 GGUF LoRA adapter.");
        if (!baseHash.Equals(ModelFingerprint(), StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException("GGUF LoRA belongs to a different base model.");
        int step = metadata.TryGetValue("nntrain.training.step", out object? stepValue)
            ? stepValue switch
            {
                uint value when value <= int.MaxValue => (int)value,
                int value when value >= 0 => value,
                _ => throw new InvalidDataException("Invalid GGUF LoRA training step.")
            }
            : 0;

        var tensors = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        foreach (GgufTensorInfo tensor in reader.Tensors)
        {
            if (!tensors.TryAdd(tensor.Name, tensor) || tensor.Type != 0 || tensor.Shape.Count != 2)
                throw new InvalidDataException($"Invalid or duplicate F32 LoRA tensor: {tensor.Name}");
        }
        if (tensors.Count == 0 || tensors.Count % 2 != 0)
            throw new InvalidDataException("GGUF LoRA tensor directory is incomplete.");
        string[] names = tensors.Keys.Select(name =>
        {
            if (name.EndsWith(".lora_a", StringComparison.Ordinal) ||
                name.EndsWith(".lora_b", StringComparison.Ordinal))
                return name[..^7];
            throw new InvalidDataException($"Unexpected GGUF LoRA tensor: {name}");
        }).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        if (tensors.Count != names.Length * 2)
            throw new InvalidDataException("GGUF LoRA is missing A/B tensors.");
        var targets = new HashSet<string>(StringComparer.Ordinal);
        var layers = new HashSet<int>();
        bool includeOutput = false;
        foreach (string name in names)
        {
            if (name == "output.weight") { includeOutput = true; continue; }
            string[] parts = name.Split('.');
            if (parts.Length != 4 || parts[0] != "blk" || parts[3] != "weight" ||
                !int.TryParse(parts[1], out int layer) || layer < 0 || layer >= Descriptor.LayerCount ||
                !_matrices.ContainsKey(name))
                throw new InvalidDataException($"LoRA target does not exist in this model: {name}");
            layers.Add(layer);
            targets.Add(parts[2]);
        }
        GgufTensorInfo first = tensors[names[0] + ".lora_a"];
        int rank = checked((int)first.Shape[1]);
        var options = new Qwen35LoraOptions
        {
            Rank = rank,
            Alpha = alpha,
            Targets = targets.Order(StringComparer.Ordinal).ToArray(),
            Layers = layers.Count is 0 || layers.Count == Descriptor.LayerCount ? null : layers.Order().ToArray(),
            IncludeOutput = includeOutput
        };
        options.Validate();
        string[] expected = _matrices.Keys.Where(name =>
        {
            if (!name.StartsWith("blk.", StringComparison.Ordinal)) return false;
            string[] parts = name.Split('.');
            return (options.Layers is null || options.Layers.Contains(int.Parse(parts[1]))) &&
                options.Targets.Contains(parts[2]);
        }).Concat(includeOutput ? ["output.weight"] : Array.Empty<string>())
            .Order(StringComparer.Ordinal).ToArray();
        if (!names.SequenceEqual(expected))
            throw new InvalidDataException("GGUF LoRA target directory does not match the base model.");

        ulong previousEnd = 0;
        foreach (GgufTensorInfo tensor in reader.Tensors.OrderBy(tensor => tensor.Offset))
        {
            ulong bytes = checked(tensor.Shape[0] * tensor.Shape[1] * sizeof(float));
            if (tensor.Offset < previousEnd)
                throw new InvalidDataException("GGUF LoRA tensor payloads overlap.");
            previousEnd = checked(tensor.Offset + bytes);
        }

        // Validate all shapes and finite A/B values before changing GPU state.
        var values = new Dictionary<string, (float[] A, float[] B)>(StringComparer.Ordinal);
        foreach (string name in names)
        {
            Matrix matrix = name == "output.weight" ? OutputMatrix : _matrices[name];
            GgufTensorInfo a = tensors[name + ".lora_a"];
            GgufTensorInfo b = tensors[name + ".lora_b"];
            if (a.Shape[0] != (ulong)matrix._inputWidth || a.Shape[1] != (ulong)rank ||
                b.Shape[0] != (ulong)rank || b.Shape[1] != (ulong)matrix.OutputWidth)
                throw new InvalidDataException($"GGUF LoRA matrix shape mismatch: {name}");
            values.Add(name, (ReadFloats(a, checked(matrix._inputWidth * rank)),
                ReadFloats(b, checked(matrix.OutputWidth * rank))));
        }
        try
        {
            AttachLora(options);
            foreach (string name in names)
            {
                (float[] a, float[] b) = values[name];
                _lora[name].RestoreState([a, b, [], [], [], []], training: false);
            }
            LoraStep = step;
            Reset();
        }
        catch
        {
            foreach (Qwen35LoraMatrix adapter in _lora.Values) adapter.Dispose();
            _lora.Clear();
            _loraOptions = null;
            LoraStep = 0;
            throw;
        }

        bool HasString(string key, string expectedValue) =>
            metadata.TryGetValue(key, out object? value) &&
            value is string text && text.Equals(expectedValue, StringComparison.Ordinal);

        float[] ReadFloats(GgufTensorInfo tensor, int count)
        {
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("GGUF LoRA loading requires a little-endian host.");
            byte[] bytes = reader.ReadTensorBytes(tensor, checked(count * sizeof(float)));
            float[] result = new float[count];
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(result);
            if (result.Any(value => !float.IsFinite(value)))
                throw new InvalidDataException($"GGUF LoRA tensor contains a non-finite value: {tensor.Name}");
            return result;
        }
    }
}
