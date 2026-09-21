using System.Diagnostics;
using System.Text.Json;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class DpoPipelineProbe
{
    internal static void Run(string modelPath, string outputDirectory, int capacity, int slots, int context = 128, int steps = 5)
    {
        outputDirectory = Path.GetFullPath(outputDirectory);
        if (Directory.Exists(outputDirectory)) throw new IOException("Use a fresh benchmark directory.");
        Directory.CreateDirectory(outputDirectory);
        var source = LoraConfiguration.Load<LoraTrainingConfiguration>(Path.GetFullPath("loss-traning-dpo.json"));
        // Same 20 fixed documents and a short context bound: isolate scheduling without touching the user's run.
        using (var file = new StreamWriter(Path.Combine(outputDirectory, "data.jsonl")))
            foreach (string text in DpoDataSource.Read(Path.GetFullPath(source.DataPath), source.Dataset, source.TextColumn, default).Take(checked(steps * 4)))
                file.WriteLine(JsonSerializer.Serialize(new { text }));
        var config = source with {
            LoraConfig = Path.GetFullPath(source.LoraConfig), TokenizerPath = Path.GetFullPath(source.TokenizerPath),
            DataPath = "data.jsonl", Dataset = "jsonl", ContextLength = context, BatchSize = 2, GradientAccumulationSteps = 2,
            GenerationSlots = slots, GenerationDeviceIndices = [0, 1], CompletedQueueCapacity = capacity,
            MaxSteps = steps, Epochs = 1, SaveEverySteps = steps, SampleEverySteps = 0, Resume = false, AutoResume = false,
            AdapterPath = "adapter.json", LossGraphPath = "loss.html" };
        config.Validate();
        string configPath = Path.Combine(outputDirectory, "config.json");
        File.WriteAllText(configPath, JsonSerializer.Serialize(config, LoraConfiguration.Json));
        using var log = new StreamWriter(Path.Combine(outputDirectory, "run.log")) { AutoFlush = true };
        using var output = new Tee(Console.Out, log);
        var watch = Stopwatch.StartNew();
        int code = DpoCommand.Run(Path.GetFullPath(modelPath), configPath, config, null, output);
        output.WriteLine($"benchmark wall={watch.Elapsed.TotalSeconds:F3}s exit={code}");
    }

    private sealed class Tee(TextWriter console, TextWriter file) : TextWriter
    {
        public override System.Text.Encoding Encoding => console.Encoding;
        public override void WriteLine(string? value) { lock (this) { console.WriteLine(value); file.WriteLine(value); } }
        public override void Write(char value) { lock (this) { console.Write(value); file.Write(value); } }
        public override void Flush() { console.Flush(); file.Flush(); }
    }
}
