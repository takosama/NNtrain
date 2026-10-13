using System.IO;
using System.Threading.Channels;
using NNtrain.Audio;

namespace NNtrain.Gui;

/// <summary>Creates and disposes one ASR stream per utterance, without owning visual state.</summary>
internal static class AudioRecognitionRunner
{
    internal static Task RecognizeFileAsync(ILocalAsrModel model, string path,
        Action<string> onPartial, Func<string, long, Task> onCompleted, CancellationToken ct) =>
        Task.Run(async () =>
        {
            using var file = File.OpenRead(path);
            float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
            var stream = model.CreateStream();
            using var lifetime = stream as IDisposable;
            for (int offset = 0; offset < samples.Length; offset += 5120)
            {
                ct.ThrowIfCancellationRequested();
                stream.Append(samples.AsSpan(offset, Math.Min(5120, samples.Length - offset)),
                    partial: onPartial, ct: ct);
            }
            await onCompleted(stream.Append([], final: true, ct: ct), stream.CacheBytes);
        }, ct);

    internal static Task RecognizeAsync(ILocalAsrModel model, ChannelReader<float[]> audio,
        Action<ILocalAsrPreviewControl?> onStreamCreated, Action<string> onPartial,
        Action<long> onSamplesProcessed, Func<string, long, Task> onCompleted,
        CancellationToken ct) => Task.Run(async () =>
        {
            var stream = model.CreateStream();
            using var lifetime = stream as IDisposable;
            onStreamCreated(stream as ILocalAsrPreviewControl);
            long samples = 0;
            await foreach (float[] chunk in audio.ReadAllAsync(ct))
            {
                float[] input = chunk;
                if (model is ParakeetCtcModel && audio.TryRead(out float[]? queued))
                {
                    // Keep every sample; compute only the latest queued preview.
                    var pending = new List<float[]> { chunk, queued };
                    while (pending.Count < 17 && audio.TryRead(out queued)) pending.Add(queued);
                    input = new float[pending.Sum(part => part.Length)];
                    int offset = 0;
                    foreach (float[] part in pending)
                    {
                        part.CopyTo(input, offset);
                        offset += part.Length;
                    }
                }
                samples += input.Length;
                if (samples > 16000L * 300) throw new InvalidOperationException("録音は5分までです。");
                // A completed, drained source has no tail requiring another preview.
                bool finalInput = model is ParakeetCtcModel && audio.Completion.IsCompletedSuccessfully;
                string text = stream.Append(input, finalInput,
                    partial: finalInput ? null : onPartial, ct: ct);
                onSamplesProcessed(samples);
                if (finalInput)
                {
                    await onCompleted(text, stream.CacheBytes);
                    return;
                }
            }
            await onCompleted(stream.Append([], true, ct: ct), stream.CacheBytes);
        }, ct);
}
