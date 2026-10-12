param(
    [string] $Model = (Join-Path (Split-Path $PSScriptRoot -Parent) 'models\huggingface\ggml-org\gemma-3-27b-it-GGUF\gemma-3-27b-it-Q8_0.gguf'),
    [string] $Executable = (Join-Path (Split-Path $PSScriptRoot -Parent) 'NNtrain.Gui\bin\Release\net10.0-windows\NNtrain.Gui.exe'),
    [string] $OutputPrefix = (Join-Path (Split-Path $PSScriptRoot -Parent) 'benchmark-results\gemma3-27b-q8-opencl-gui-20261012')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path -LiteralPath $Model)) { throw "Model not found: $Model" }
if (-not (Test-Path -LiteralPath $Executable)) { throw "GUI executable not found: $Executable" }

$listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
$listener.Start()
$port = ([System.Net.IPEndPoint] $listener.LocalEndpoint).Port
$listener.Stop()
$address = "http://127.0.0.1:$port"
$tokenBytes = New-Object byte[] 32
$random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
try { $random.GetBytes($tokenBytes) } finally { $random.Dispose() }
$token = [BitConverter]::ToString($tokenBytes).Replace('-', '')
$headers = @{ Authorization = "Bearer $token" }

$env:NNTRAIN_GUI_SERVER_TOKEN = $token
Remove-Item Env:GGML_OPENCL_PLATFORM -ErrorAction SilentlyContinue
Remove-Item Env:GGML_OPENCL_DEVICE -ErrorAction SilentlyContinue
$process = $null
try {
    $process = Start-Process -FilePath $Executable -ArgumentList @('--server', '--port', "$port") `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput "$OutputPrefix.server.log" `
        -RedirectStandardError "$OutputPrefix.server-error.log"
    [Environment]::SetEnvironmentVariable('NNTRAIN_GUI_SERVER_TOKEN', $null, 'Process')

    $ready = $false
    $lastHealthError = ''
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if ($process.HasExited) { throw "GUI server exited with code $($process.ExitCode)" }
        try {
            $health = Invoke-RestMethod -Uri "$address/health" -Headers $headers -TimeoutSec 2
            if ($health.status -eq 'ok') { $ready = $true; break }
        } catch {
            $lastHealthError = $_.Exception.Message
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $ready) { throw "GUI server did not become ready: $lastHealthError" }

    $load = @{
        model = [IO.Path]::GetFullPath($Model)
        devices = @()
        gemma_backend = 'opencl'
    } | ConvertTo-Json -Depth 5 -Compress
    $state = Invoke-RestMethod -Uri "$address/internal/load" -Method Post -Headers $headers `
        -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($load)) -TimeoutSec 300
    if (-not $state.is_loaded -or $state.gemma_backend -ne 'opencl') {
        throw "Unexpected model state: $($state | ConvertTo-Json -Compress)"
    }

    $chat = @{
        model = [IO.Path]::GetFullPath($Model)
        devices = @()
        gemma_backend = 'opencl'
        messages = @(@{ role = 'user'; content = '2足す3はいくつですか。数字だけで答えてください。' })
        max_tokens = 8
        temperature = 0
        top_p = 1
        top_k = 1
        think = $false
        stream = $false
    } | ConvertTo-Json -Depth 8 -Compress
    $response = Invoke-RestMethod -Uri "$address/v1/chat/completions" -Method Post `
        -Headers $headers -ContentType 'application/json; charset=utf-8' `
        -Body ([Text.Encoding]::UTF8.GetBytes($chat)) -TimeoutSec 300
    $response | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$OutputPrefix.response.json" -Encoding utf8
    $answer = [string] $response.choices[0].message.content
    if ($answer.Trim() -ne '5') { throw "Unexpected answer: $answer" }
    Write-Output "OpenCL GUI API response: $answer; $OutputPrefix.response.json"
} finally {
    if ($null -ne $process -and -not $process.HasExited) {
        try {
            Invoke-RestMethod -Uri "$address/internal/shutdown" -Method Post -Headers $headers `
                -ContentType 'application/json' -Body '{}' -TimeoutSec 10 | Out-Null
        } catch { }
        if (-not $process.WaitForExit(10000)) { Stop-Process -Id $process.Id -Force }
    }
    if ($null -ne $process) { $process.Dispose() }
    [Environment]::SetEnvironmentVariable('NNTRAIN_GUI_SERVER_TOKEN', $null, 'Process')
}
