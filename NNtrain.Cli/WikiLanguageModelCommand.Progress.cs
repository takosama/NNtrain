using System.Diagnostics;
using System.Globalization;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    /// <summary>
    /// Tracks wall time and overall training progress for the periodic log
    /// line.
    /// </summary>
    /// <remarks>
    /// The remaining-time estimate deliberately measures only the progress
    /// this process has made. A run resumed from a checkpoint starts with the
    /// stopwatch at zero but the progress fraction already well above zero, so
    /// dividing total elapsed time by total progress would report a remaining
    /// time far shorter than the truth. The first reported fraction becomes the
    /// anchor instead, which costs one log line before an estimate appears.
    /// </remarks>
    internal sealed class TrainingProgress
    {
        private readonly Stopwatch _timer = Stopwatch.StartNew();
        private double? _anchorFraction;
        private TimeSpan _anchorElapsed;

        public TimeSpan Elapsed => _timer.Elapsed;

        /// <summary>
        /// Formats the completion percentage, elapsed wall time, and remaining
        /// time for an overall progress fraction in [0, 1].
        /// </summary>
        public string Describe(double fraction)
        {
            TimeSpan elapsed = _timer.Elapsed;
            if (!double.IsFinite(fraction))
                return $"progress = ?, elapsed = {FormatElapsed(elapsed)}";

            fraction = Math.Clamp(fraction, 0d, 1d);
            if (_anchorFraction is null)
            {
                _anchorFraction = fraction;
                _anchorElapsed = elapsed;
            }

            string text =
                $"progress = {fraction * 100d:F2}%, " +
                $"elapsed = {FormatElapsed(elapsed)}";

            double advanced = fraction - _anchorFraction.Value;
            double measured = (elapsed - _anchorElapsed).TotalSeconds;
            if (fraction >= 1d || advanced <= 0d || measured <= 0d)
                return $"{text}, ETA = --";

            double remaining = (1d - fraction) / advanced * measured;
            if (!double.IsFinite(remaining)
                || remaining < 0d
                || remaining > TimeSpan.MaxValue.TotalSeconds / 2d)
            {
                return $"{text}, ETA = --";
            }

            var span = TimeSpan.FromSeconds(remaining);
            string finish = DateTime.Now
                .AddSeconds(remaining)
                .ToString("MM/dd HH:mm", CultureInfo.InvariantCulture);
            return $"{text}, ETA = {FormatElapsed(span)} (~{finish})";
        }
    }
}
