namespace NNtrain;

/// <summary>
/// Controls whether forward operations record automatic-differentiation history.
/// </summary>
public static class AutogradContext
{
    private static readonly AsyncLocal<int> NoGradDepth = new();

    internal static bool IsRecordingEnabled => NoGradDepth.Value == 0;

    /// <summary>
    /// Disables graph recording until the returned scope is disposed.
    /// </summary>
    public static IDisposable NoGrad()
    {
        NoGradDepth.Value++;
        return new NoGradScope();
    }

    private sealed class NoGradScope : IDisposable
    {
        private bool _disposed;

        public void Dispose()
        {
            if (_disposed)
                return;

            int depth = NoGradDepth.Value;
            if (depth <= 0)
            {
                throw new InvalidOperationException(
                    "NoGrad scopes must be disposed in the execution context where they are active.");
            }

            NoGradDepth.Value = depth - 1;
            _disposed = true;
        }
    }
}
