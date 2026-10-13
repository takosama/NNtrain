namespace NNtrain;

internal static class TrainingResumePolicy
{
    internal static bool Resolve(
        bool explicitResume,
        bool autoResume,
        TrainingRunGuard run,
        string checkpointPath,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(run);
        ArgumentNullException.ThrowIfNull(output);
        if (explicitResume)
            return true;
        if (!autoResume)
            return false;
        if (!run.WasInterrupted)
        {
            output.WriteLine(
                "auto-resume = no interrupted training run detected; " +
                "starting from the configured initial state");
            return false;
        }
        if (!File.Exists(checkpointPath))
        {
            output.WriteLine(
                $"auto-resume = interrupted run marker found at " +
                $"{run.MarkerPath}, but checkpoint is missing; starting " +
                "from the configured initial state");
            return false;
        }

        output.WriteLine(
            $"auto-resume = interrupted training detected; restoring " +
            $"latest checkpoint {Path.GetFullPath(checkpointPath)}");
        return true;
    }

}
