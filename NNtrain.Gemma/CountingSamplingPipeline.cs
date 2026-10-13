using System.Diagnostics;
using LLama.Native;
using LLama.Sampling;

namespace NNtrain.Gemma;

internal sealed class CountingSamplingPipeline(DefaultSamplingPipeline inner,
    Stopwatch timer) : ISamplingPipeline
{
    public int SampledTokens { get; private set; }
    public double? FirstSampleMilliseconds { get; private set; }
    public bool LastSampleWasEndOfGeneration { get; private set; }

    public LLamaToken Sample(SafeLLamaContextHandle context, int index)
    {
        LLamaToken token = inner.Sample(context, index);
        SampledTokens++;
        FirstSampleMilliseconds ??= timer.Elapsed.TotalMilliseconds;
        LastSampleWasEndOfGeneration = token.IsEndOfGeneration(context.Vocab);
        return token;
    }

    public void Apply(SafeLLamaContextHandle context, LLamaTokenDataArray data)
        => inner.Apply(context, data);

    public void Accept(LLamaToken token) => inner.Accept(token);

    public void Reset() => inner.Reset();

    // The outer GenerateAsync scope owns and disposes the native pipeline.
    public void Dispose() { }
}
