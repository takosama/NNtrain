using System.Diagnostics;
using System.Runtime.ExceptionServices;

namespace NNtrain;

/// <summary>Reports CPU/disk cursor replay, including time blocked inside MoveNext.</summary>
internal sealed class ResumeDocumentProgress : IDisposable
{
    private readonly TextWriter _output;
    private readonly long _total;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly object _sync = new();
    private readonly Timer _timer;
    private long _completed;
    private bool _finished;
    private ExceptionDispatchInfo? _outputFailure;

    internal ResumeDocumentProgress(long total, TextWriter output, TimeSpan? interval = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(total);
        ArgumentNullException.ThrowIfNull(output);
        _total = total;
        _output = output;
        output.WriteLine($"resume data cursor = replaying {total:N0} previously consumed documents " +
            "in the original shuffle order; CPU/disk work, GPU idle is expected (no optimizer updates)");
        output.Flush();
        TimeSpan period = interval ?? TimeSpan.FromSeconds(5);
        _timer = new Timer(_ => Report(), null, period, period);
    }

    internal void Advance()
    {
        Volatile.Read(ref _outputFailure)?.Throw();
        Interlocked.Increment(ref _completed);
    }

    private void Report()
    {
        lock (_sync)
        {
            if (_finished || _outputFailure is not null) return;
            try
            {
                long done = Interlocked.Read(ref _completed);
                double seconds = _clock.Elapsed.TotalSeconds;
                string eta = done == 0 ? "--" : $"{seconds * (_total - done) / done:F0} sec";
                _output.WriteLine($"resume data cursor = {done:N0}/{_total:N0} " +
                    $"({100d * done / _total:F2}%), elapsed = {seconds:F1} sec, ETA = {eta}" +
                    (done == 0 ? "; reading corpus metadata / filling shuffle buffer" : ""));
                _output.Flush();
            }
            catch (Exception error)
            {
                // Do not crash the process on a thread-pool callback. Surface
                // output errors on the foreground iterator instead.
                Volatile.Write(ref _outputFailure, ExceptionDispatchInfo.Capture(error));
            }
        }
    }

    internal void Complete()
    {
        lock (_sync)
        {
            _outputFailure?.Throw();
            _finished = true;
            _output.WriteLine($"resume data cursor restored = {_completed:N0} documents, " +
                $"{_clock.Elapsed.TotalSeconds:F1} sec; continuing from saved token buffer " +
                "(first CUDA step may compile/capture)");
            _output.Flush();
        }
    }

    public void Dispose()
    {
        _timer.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}
