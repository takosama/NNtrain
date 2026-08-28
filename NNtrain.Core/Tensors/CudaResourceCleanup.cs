namespace NNtrain;

/// <summary>Cold-path native cleanup that always attempts every release.</summary>
internal static class CudaResourceCleanup
{
    internal static void RunAll(
        string message,
        params Action[] releases)
        => RunAll(message, (IEnumerable<Action>)releases);

    internal static void RunAll(
        string message,
        IEnumerable<Action> releases)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message);
        ArgumentNullException.ThrowIfNull(releases);
        List<Exception>? failures = null;
        foreach (Action release in releases)
        {
            try
            {
                release();
            }
            catch (AggregateException aggregate)
            {
                (failures ??= []).AddRange(
                    aggregate.Flatten().InnerExceptions);
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (failures is not null)
            throw new AggregateException(message, failures);
    }

    internal static void RunAllNoThrow(IEnumerable<Action> releases)
    {
        try
        {
            RunAll("Finalizer resource cleanup failed.", releases);
        }
        catch
        {
            // Finalizers must never terminate the process. Explicit Dispose
            // surfaces the same failures as an AggregateException.
        }
    }
}
