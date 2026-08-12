namespace NNtrain;

/// <summary>
/// The executable entry point intentionally mirrors a PyTorch program: parse
/// configuration, seed torch, then hand control to the training application.
/// </summary>
internal static partial class Program
{
    private static int Main(string[] args) => main(args);

    internal static int main(string[] args)
        => run(args, Console.Out, Console.Error, open_loss_graph: true);

    internal static int run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool open_loss_graph = false)
        => Run(
            args,
            output,
            error,
            open_loss_graph);
}
