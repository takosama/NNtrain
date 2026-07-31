namespace NNtrain;

class Program
{
    static int Main(string[] args)
    {
        return Run(args, Console.Out, Console.Error);
    }

    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);

        if (args.Length != 2
            || !string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase))
        {
            error.WriteLine(
                "Usage: NNtrain.Cli --config <training-config.json>");
            return 1;
        }

        try
        {
            TrainingConfiguration configuration =
                TrainingConfiguration.Load(args[1]);
            TrainingComponents components =
                TrainingComposition.Create(configuration);
            components.Trainer.Run(result => WriteMetrics(output, result));
            return 0;
        }
        catch (Exception exception) when (
            exception is IOException
            or System.Text.Json.JsonException
            or ArgumentException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }

    private static void WriteMetrics(
        TextWriter output,
        TrainingEpochResult result)
    {
        output.WriteLine(
            $"epoch {result.Epoch}, " +
            $"train loss = {result.Training.Loss:F6}, " +
            $"train acc = {result.Training.Accuracy * 100f:F2}%, " +
            $"eval loss = {result.Evaluation.Loss:F6}, " +
            $"eval acc = {result.Evaluation.Accuracy * 100f:F2}%, " +
            $"train time = {result.Training.Elapsed.TotalSeconds:F2} sec, " +
            $"eval time = {result.Evaluation.Elapsed.TotalSeconds:F2} sec");
    }
}
