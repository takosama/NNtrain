using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;

namespace NNtrain.Gui;

public partial class App : Application
{
    protected override async void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        try
        {
            if (e.Args.Length > 0 && e.Args[0] == "--server")
            {
                ShutdownMode = ShutdownMode.OnExplicitShutdown;
                await RunServerAsync(e.Args);
                Shutdown();
                return;
            }

            GuiLaunchOptions options = GuiLaunchOptions.Parse(e.Args);
            var window = new MainWindow(options);
            MainWindow = window;
            window.Show();
        }
        catch (Exception ex)
        {
            if (e.Args.Length > 0 && e.Args[0] == "--server")
                Console.Error.WriteLine(ex);
            else
                MessageBox.Show(ex.Message, "NNtrain 起動エラー", MessageBoxButton.OK, MessageBoxImage.Error);
            Shutdown(1);
        }
    }

    private static async Task RunServerAsync(string[] args)
    {
        AttachConsole();
        int? port = null;
        int? parentPid = null;
        string? lora = null;
        for (int i = 1; i < args.Length; i++)
        {
            if (i + 1 >= args.Length)
                throw new ArgumentException($"{args[i]} の値がありません。");
            string value = args[++i];
            switch (args[i - 1])
            {
                case "--port":
                    if (!int.TryParse(value, out int parsedPort) || parsedPort is < 1 or > 65535)
                        throw new ArgumentException("--port は 1 ～ 65535 で指定してください。");
                    port = parsedPort;
                    break;
                case "--parent-pid":
                    if (!int.TryParse(value, out int parsedPid) || parsedPid < 1)
                        throw new ArgumentException("--parent-pid が不正です。");
                    parentPid = parsedPid;
                    break;
                case "--lora":
                    RejectParentTraversal(value);
                    lora = Path.GetFullPath(value);
                    break;
                default:
                    throw new ArgumentException($"不明な引数: {args[i - 1]}");
            }
        }
        if (port is null)
            throw new ArgumentException("サーバーには --port が必要です。");

        string? suppliedToken = Environment.GetEnvironmentVariable(LocalOpenAiServer.TokenEnvironmentVariable);
        await using var server = new LocalOpenAiServer(suppliedToken);
        using var cancellation = new CancellationTokenSource();
        Uri address = await server.StartAsync(port.Value, lora, cancellation.Token);
        Console.WriteLine($"NNtrain OpenAI API server: {address}");
        if (suppliedToken is null) Console.WriteLine($"Session Authorization: Bearer {server.AuthenticationToken}");
        if (parentPid is int pid)
        {
            using Process parent = Process.GetProcessById(pid);
            Task parentExit = parent.WaitForExitAsync(cancellation.Token);
            Task shutdown = server.WaitForShutdownAsync(cancellation.Token);
            await Task.WhenAny(parentExit, shutdown);
            cancellation.Cancel();
        }
        else
            await server.WaitForShutdownAsync(cancellation.Token);
    }

    private static void AttachConsole()
    {
        if (!AllocConsole()) return;
        Console.OutputEncoding = Encoding.UTF8;
        Console.SetOut(new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true });
        Console.SetError(new StreamWriter(Console.OpenStandardError(), new UTF8Encoding(false)) { AutoFlush = true });
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AllocConsole();

    internal static void RejectParentTraversal(string path)
    {
        if (string.IsNullOrWhiteSpace(path) ||
            path.Replace('\\', '/').Split('/').Any(segment => segment == ".."))
            throw new ArgumentException("パスに ../ または ..\\ は指定できません。");
    }
}

internal sealed record GuiLaunchOptions(
    string? ModelPath, string? LoraPath, string? TopP, string? TopK,
    string? Temperature, string? MaxTokens, bool Stream, bool Think)
{
    internal static GuiLaunchOptions Parse(string[] args)
    {
        string? model = null, lora = null, topP = null, topK = null;
        string? temperature = null, maxTokens = null;
        bool stream = true, think = true;
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            if (i + 1 >= args.Length)
                throw new ArgumentException($"{option} の値がありません。");
            string value = args[++i];
            switch (option)
            {
                case "--model": App.RejectParentTraversal(value); model = Path.GetFullPath(value); break;
                case "--lora": App.RejectParentTraversal(value); lora = Path.GetFullPath(value); break;
                case "--top_p": case "--top-p": topP = value; break;
                case "--top_k": case "--top-k": topK = value; break;
                case "--temperature": case "--tempreture": temperature = value; break;
                case "--maxtokens": case "--max_tokens": maxTokens = value; break;
                case "--stream": stream = ParseSwitch(option, value); break;
                case "--think": think = ParseSwitch(option, value); break;
                default: throw new ArgumentException($"不明な引数: {option}");
            }
        }
        if (topP is not null && (!float.TryParse(topP, NumberStyles.Float, CultureInfo.InvariantCulture, out float p) || !float.IsFinite(p) || p is <= 0 or > 1))
            throw new ArgumentException("--top_p は 0 より大きく 1 以下で指定してください。");
        if (topK is not null && (!int.TryParse(topK, out int k) || k is < 1 or > 256))
            throw new ArgumentException("--top_k は 1 ～ 256 で指定してください。");
        if (temperature is not null && (!float.TryParse(temperature, NumberStyles.Float, CultureInfo.InvariantCulture, out float t) || !float.IsFinite(t) || t is < 0 or > 2))
            throw new ArgumentException("--temperature は 0 ～ 2 で指定してください。");
        if (maxTokens is not null && (!int.TryParse(maxTokens, out int max) || max is < 1 or > 8192))
            throw new ArgumentException("--maxtokens は 1 ～ 8192 で指定してください。");
        return new GuiLaunchOptions(model, lora, topP, topK, temperature, maxTokens, stream, think);
    }

    private static bool ParseSwitch(string option, string value) => value.ToLowerInvariant() switch
    {
        "on" => true,
        "off" => false,
        _ => throw new ArgumentException($"{option} は on または off で指定してください。")
    };
}
