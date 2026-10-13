using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Security.Cryptography;

namespace NNtrain.Gui;

/// <summary>Owns only the child started for this window, including its session credential.</summary>
internal sealed class LocalInferenceServerProcess(string? loraPath, string tokenEnvironmentVariable) : IDisposable
{
    private Process? _process;
    private int? _port;

    internal string AuthenticationToken { get; } = Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
    internal bool IsReady { get; private set; }
    internal bool IsRunning => _process is { HasExited: false };

    internal async Task StartAsync(LocalInferenceClient client)
    {
        if (_port is null)
        {
            var listener = new TcpListener(IPAddress.Loopback, 0);
            listener.Start();
            _port = ((IPEndPoint)listener.LocalEndpoint).Port;
            listener.Stop();
            client.SetAddress(new Uri($"http://127.0.0.1:{_port}/"));
        }
        string executable = Environment.ProcessPath
            ?? throw new InvalidOperationException("実行ファイルの場所を取得できません。");
        _process = Process.Start(CreateStartInfo(executable, AppContext.BaseDirectory, _port.Value,
            Environment.ProcessId, loraPath, tokenEnvironmentVariable, AuthenticationToken))
            ?? throw new InvalidOperationException("推論サーバーを起動できませんでした。");
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        while (!timeout.IsCancellationRequested)
        {
            if (_process.HasExited)
                throw new InvalidOperationException($"推論サーバーが終了しました（コード {_process.ExitCode}）。");
            try
            {
                using var probe = new CancellationTokenSource(TimeSpan.FromSeconds(1));
                if (await client.ProbeAsync(probe.Token)) break;
            }
            catch (HttpRequestException) { }
            catch (OperationCanceledException) { }
            await Task.Delay(200, timeout.Token);
        }
        timeout.Token.ThrowIfCancellationRequested();
        IsReady = true;
    }

    internal static ProcessStartInfo CreateStartInfo(string executable, string baseDirectory, int port,
        int parentPid, string? adapterPath, string tokenEnvironmentVariable, string token)
    {
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            WindowStyle = ProcessWindowStyle.Normal
        };
        start.Environment[tokenEnvironmentVariable] = token;
        if (Path.GetFileNameWithoutExtension(executable).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            start.ArgumentList.Add(Path.Combine(baseDirectory, "NNtrain.Gui.dll"));
        start.ArgumentList.Add("--server");
        start.ArgumentList.Add("--port");
        start.ArgumentList.Add(port.ToString(CultureInfo.InvariantCulture));
        if (adapterPath is { } lora)
        {
            start.ArgumentList.Add("--lora");
            start.ArgumentList.Add(lora);
        }
        start.ArgumentList.Add("--parent-pid");
        start.ArgumentList.Add(parentPid.ToString(CultureInfo.InvariantCulture));
        return start;
    }

    internal Task WaitForExitAsync(CancellationToken ct) => _process?.WaitForExitAsync(ct) ?? Task.CompletedTask;

    internal void ResetExitedProcess()
    {
        if (IsRunning) throw new InvalidOperationException("実行中の推論サーバーはリセットできません。");
        IsReady = false;
        _process?.Dispose();
        _process = null;
    }

    internal void EnsureReady()
    {
        if (!IsReady || !IsRunning)
            throw new InvalidOperationException("推論サーバーに接続できません。");
    }

    internal async Task StopAsync(LocalInferenceClient client)
    {
        try
        {
            if (IsReady && IsRunning)
            {
                using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(2));
                await client.PostAsync("internal/shutdown", new { }, timeout.Token);
                try { await WaitForExitAsync(timeout.Token); }
                catch (OperationCanceledException) { }
            }
        }
        catch { /* The owned child is stopped below if graceful shutdown fails. */ }
        finally { Dispose(); }
    }

    public void Dispose()
    {
        if (_process is { HasExited: false } child)
        {
            try { child.Kill(entireProcessTree: true); }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
        }
        _process?.Dispose();
        _process = null;
        IsReady = false;
    }
}
