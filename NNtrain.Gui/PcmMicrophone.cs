using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace NNtrain.Gui;

/// <summary>Explicit-start WinMM capture. Constructor is called only by the Record command.</summary>
internal sealed class PcmMicrophone : IDisposable
{
    private readonly object _gate = new();
    private readonly Dictionary<nint, nint> _buffers = [];
    private readonly WaveCallback _callback;
    private readonly Channel<short[]> _raw = Channel.CreateBounded<short[]>(new BoundedChannelOptions(16)
        { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
    private readonly Channel<float[]> _channel = Channel.CreateBounded<float[]>(new BoundedChannelOptions(16)
        { SingleReader = true, FullMode = BoundedChannelFullMode.Wait });
    private nint _device;
    private bool _stopping, _disposed;
    private int _queued;
    internal ChannelReader<float[]> Audio => _channel.Reader;
    internal int CaptureSampleRate { get; private set; }
    internal int CaptureChannels { get; private set; }

    internal PcmMicrophone()
    {
        _callback = OnWaveMessage;
        var format = new WaveFormat { FormatTag = 1, Channels = 2, SamplesPerSec = 48000,
            AverageBytesPerSec = 192000, BlockAlign = 4, BitsPerSample = 16 };
        // WAVE_FORMAT_QUERY opens no device and captures no samples.
        if (waveInOpen(out _, uint.MaxValue, ref format, _callback, 0, 0x30001) != 0)
        {
            format.Channels = 1; format.BlockAlign = 2; format.AverageBytesPerSec = 96000;
            if (waveInOpen(out _, uint.MaxValue, ref format, _callback, 0, 0x30001) != 0)
            { format.SamplesPerSec = 16000; format.AverageBytesPerSec = 32000; }
        }
        CaptureSampleRate = (int)format.SamplesPerSec;
        CaptureChannels = format.Channels;
        int bufferBytes = checked((int)format.AverageBytesPerSec * 320 / 1000);
        _ = Task.Run(ConvertAudioAsync);
        try
        {
            Check(waveInOpen(out _device, uint.MaxValue, ref format, _callback, 0, 0x30000), "マイクを開く");
            for (int i = 0; i < 4; i++)
            {
                nint data = Marshal.AllocHGlobal(bufferBytes);
                nint header = Marshal.AllocHGlobal(Marshal.SizeOf<WaveHeader>());
                Marshal.StructureToPtr(new WaveHeader { Data = data, BufferLength = (uint)bufferBytes }, header, false);
                _buffers.Add(header, data);
                Check(waveInPrepareHeader(_device, header, (uint)Marshal.SizeOf<WaveHeader>()), "バッファを準備");
                QueueBuffer(header);
            }
            Check(waveInStart(_device), "録音を開始");
        }
        catch { Dispose(); throw; }
    }

    private void QueueBuffer(nint header)
    {
        _queued++;
        uint result = waveInAddBuffer(_device, header, (uint)Marshal.SizeOf<WaveHeader>());
        if (result != 0) { _queued--; Check(result, "バッファを登録"); }
    }

    private void OnWaveMessage(nint device, uint message, nint instance, nint pointer, nint reserved)
    {
        if (message != 0x3C0) return;
        lock (_gate)
        {
            if (_disposed || !_buffers.ContainsKey(pointer)) return;
            _queued--;
            var header = Marshal.PtrToStructure<WaveHeader>(pointer);
            if (header.BytesRecorded > 0)
            {
                var pcm = new short[header.BytesRecorded / 2];
                Marshal.Copy(header.Data, pcm, 0, pcm.Length);
                if (!_raw.Writer.TryWrite(pcm))
                {
                    _stopping = true;
                    _raw.Writer.TryComplete(new InvalidOperationException("認識が録音速度に追いつきません。WAVファイルを使用してください。"));
                }
            }
            if (_stopping) { if (_queued == 0) _raw.Writer.TryComplete(); }
            else ThreadPool.QueueUserWorkItem(_ => Requeue(pointer));
        }
    }

    private async Task ConvertAudioAsync()
    {
        try
        {
            var converter = new NNtrain.Audio.Pcm16StreamResampler(CaptureSampleRate, CaptureChannels);
            await foreach (short[] pcm in _raw.Reader.ReadAllAsync())
            {
                float[] samples = converter.Append(pcm);
                if (samples.Length > 0 && !_channel.Writer.TryWrite(samples))
                    throw new InvalidOperationException("Audio recognition queue overflow.");
            }
            float[] tail = converter.Append([], final: true);
            if (tail.Length > 0 && !_channel.Writer.TryWrite(tail))
                throw new InvalidOperationException("Audio finalization queue overflow.");
            _channel.Writer.TryComplete();
        }
        catch (Exception ex) { _raw.Writer.TryComplete(ex); _channel.Writer.TryComplete(ex); }
    }

    private void Requeue(nint pointer)
    {
        lock (_gate)
        {
            if (_stopping || _disposed) return;
            try { QueueBuffer(pointer); }
            catch (Exception ex) { _stopping = true; _raw.Writer.TryComplete(ex); }
        }
    }

    internal void Stop()
    {
        lock (_gate)
        {
            if (_stopping || _disposed) return;
            _stopping = true;
        }
        // Reset returns the partial final buffer. Complete only after all queued callbacks deliver it.
        Check(waveInStop(_device), "録音を停止");
        Check(waveInReset(_device), "録音の残りを回収");
        lock (_gate) { if (_queued == 0) _raw.Writer.TryComplete(); }
    }

    public void Dispose()
    {
        lock (_gate) { if (_disposed) return; _stopping = true; }
        if (_device != 0) { waveInStop(_device); waveInReset(_device); }
        lock (_gate)
        {
            _disposed = true;
            foreach (var (header, data) in _buffers)
            {
                if (_device != 0) waveInUnprepareHeader(_device, header, (uint)Marshal.SizeOf<WaveHeader>());
                Marshal.FreeHGlobal(header); Marshal.FreeHGlobal(data);
            }
            _buffers.Clear();
            if (_device != 0) { waveInClose(_device); _device = 0; }
            _raw.Writer.TryComplete();
        }
    }

    private static void Check(uint result, string operation)
    { if (result != 0) throw new InvalidOperationException($"{operation}に失敗しました (WinMM {result})。"); }
    [StructLayout(LayoutKind.Sequential, Pack = 2)] private struct WaveFormat
    { public ushort FormatTag, Channels; public uint SamplesPerSec, AverageBytesPerSec; public ushort BlockAlign, BitsPerSample, ExtraSize; }
    [StructLayout(LayoutKind.Sequential)] private struct WaveHeader
    { public nint Data; public uint BufferLength, BytesRecorded; public nint User; public uint Flags, Loops; public nint Next, Reserved; }
    [UnmanagedFunctionPointer(CallingConvention.Winapi)] private delegate void WaveCallback(nint device, uint message, nint instance, nint header, nint reserved);
    [DllImport("winmm.dll")] private static extern uint waveInOpen(out nint device, uint deviceId, ref WaveFormat format, WaveCallback callback, nint instance, uint flags);
    [DllImport("winmm.dll")] private static extern uint waveInPrepareHeader(nint device, nint header, uint size);
    [DllImport("winmm.dll")] private static extern uint waveInUnprepareHeader(nint device, nint header, uint size);
    [DllImport("winmm.dll")] private static extern uint waveInAddBuffer(nint device, nint header, uint size);
    [DllImport("winmm.dll")] private static extern uint waveInStart(nint device);
    [DllImport("winmm.dll")] private static extern uint waveInStop(nint device);
    [DllImport("winmm.dll")] private static extern uint waveInReset(nint device);
    [DllImport("winmm.dll")] private static extern uint waveInClose(nint device);
}
