namespace NNtrain;

/// <summary>
/// The executable entry point intentionally mirrors a PyTorch program: parse
/// configuration, seed torch, then hand control to the training application.
/// </summary>
internal static class Program
{
    private static int Main(string[] args) => main(args);

    internal static int main(string[] args)
        => run(args, Console.Out, Console.Error, open_loss_graph: true);

    internal static int run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool open_loss_graph = false)
        => TorchTrainingApplication.Run(
            args,
            output,
            error,
            open_loss_graph);

    // Compatibility entry points for existing hosts and tests.
    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false)
        => run(args, output, error, openLossGraph);

    internal static int DivideRoundUp(int value, int divisor)
        => TorchTrainingApplication.DivideRoundUp(value, divisor);

    internal static string FindDefaultConfiguration()
        => TorchTrainingApplication.FindDefaultConfiguration();

    internal static IOptimizer CreateOptimizer(
        TransformerClassifier model,
        TrainingConfiguration configuration)
        => TorchTrainingApplication.CreateOptimizer(model, configuration);

    internal static float CalculateLearningRateFactor(
        int epoch,
        int totalEpochs,
        int warmupEpochs,
        float minimumRatio)
        => LinearWarmupCosineLRScheduler.CalculateFactor(
            epoch,
            totalEpochs,
            warmupEpochs,
            minimumRatio);

    internal static LearningRates SetScheduledLearningRates(
        IOptimizer optimizer,
        TrainingConfiguration configuration,
        int epoch)
    {
        TorchTrainingApplication.LearningRates rates =
            TorchTrainingApplication.SetScheduledLearningRates(
                optimizer,
                configuration,
                epoch);
        return new LearningRates(rates.Primary, rates.Auxiliary);
    }

    internal readonly record struct LearningRates(
        float Primary,
        float? Auxiliary);
}
