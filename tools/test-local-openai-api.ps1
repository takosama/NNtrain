<#
.SYNOPSIS
Checks the local OpenAI-compatible API without loading a model.
.EXAMPLE
pwsh -File .\tools\test-local-openai-api.ps1
.EXAMPLE
pwsh -File .\tools\test-local-openai-api.ps1 -ExePath .\NNtrain.Gui\bin\Release\net10.0-windows\NNtrain.Gui.exe
#>
param(
    [string] $ExePath
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Net.Http

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($ExePath)) {
    $project = Join-Path $repoRoot 'NNtrain.Gui\NNtrain.Gui.csproj'
    & dotnet build $project -c Debug --no-restore --nologo --verbosity quiet
    if ($LASTEXITCODE -ne 0) {
        throw "GUI build failed (exit code $LASTEXITCODE). Restore project packages first if needed."
    }
    $ExePath = Join-Path $repoRoot 'NNtrain.Gui\bin\Debug\net10.0-windows\NNtrain.Gui.exe'
}
$ExePath = (Resolve-Path -LiteralPath $ExePath).Path

$logDir = Join-Path ([System.IO.Path]::GetTempPath()) ("nntrain-api-smoke-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $logDir | Out-Null
$client = [System.Net.Http.HttpClient]::new()
$client.Timeout = [TimeSpan]::FromSeconds(10)
$serverProcess = $null
$baseUri = $null
$passed = 0

function New-FreePort {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try { return ([System.Net.IPEndPoint] $listener.LocalEndpoint).Port }
    finally { $listener.Stop() }
}

function Start-TestProcess([string] $tag, [int] $port, [string[]] $extraArgs) {
    $arguments = @('--server', '--port', [string] $port) + $extraArgs
    $stdout = Join-Path $logDir "$tag.stdout.log"
    $stderr = Join-Path $logDir "$tag.stderr.log"
    $process = Start-Process -FilePath $ExePath -ArgumentList $arguments -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    return [pscustomobject] @{ Process = $process; Stdout = $stdout; Stderr = $stderr }
}

function Stop-TestProcess([System.Diagnostics.Process] $process) {
    if ($null -ne $process -and -not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit(5000) | Out-Null
    }
    if ($null -ne $process) { $process.Dispose() }
}

function Get-ProcessLogs($started) {
    $lines = @()
    foreach ($file in @($started.Stdout, $started.Stderr)) {
        if (Test-Path -LiteralPath $file) {
            $lines += Get-Content -LiteralPath $file -ErrorAction SilentlyContinue
        }
    }
    return ($lines -join [Environment]::NewLine)
}

function Send-ApiRequest([string] $method, [string] $path, [object] $body = $null) {
    $request = [System.Net.Http.HttpRequestMessage]::new(
        [System.Net.Http.HttpMethod]::new($method), "$baseUri$path")
    if ($null -ne $body) {
        $json = ConvertTo-Json -InputObject $body -Depth 12 -Compress
        $request.Content = [System.Net.Http.StringContent]::new(
            $json, [System.Text.Encoding]::UTF8, 'application/json')
    }
    try {
        $response = $client.SendAsync($request).GetAwaiter().GetResult()
        try {
            $content = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
            $parsed = if ([string]::IsNullOrWhiteSpace($content)) { $null }
                else { ConvertFrom-Json -InputObject $content }
            return [pscustomobject] @{
                Status = [int] $response.StatusCode
                Json = $parsed
                Content = $content
            }
        }
        finally { $response.Dispose() }
    }
    finally { $request.Dispose() }
}

function Assert-Status([string] $name, $response, [int] $expected) {
    if ($response.Status -ne $expected) {
        throw "$name returned HTTP $($response.Status), expected $expected. Body: $($response.Content)"
    }
    $script:passed++
    Write-Host "PASS $name (HTTP $expected)"
}

try {
    foreach ($traversal in @('../adapter.bin', '..\adapter.bin')) {
        $tag = if ($traversal.Contains('/')) { 'lora-forward' } else { 'lora-backslash' }
        $started = Start-TestProcess $tag (New-FreePort) @('--lora', $traversal)
        try {
            if (-not $started.Process.WaitForExit(5000)) {
                throw "CLI accepted forbidden --lora path '$traversal'."
            }
            if ($started.Process.ExitCode -eq 0) {
                throw "CLI exited successfully for forbidden --lora path '$traversal'."
            }
            $passed++
            Write-Host "PASS --lora $traversal rejected"
        }
        finally { Stop-TestProcess $started.Process }
    }

    $port = New-FreePort
    $baseUri = "http://127.0.0.1:$port"
    $serverProcess = Start-TestProcess 'server' $port @()
    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($serverProcess.Process.HasExited) {
            throw "Server exited before becoming ready. $(Get-ProcessLogs $serverProcess)"
        }
        try {
            $health = Send-ApiRequest 'GET' '/health'
            if ($health.Status -eq 200) { break }
        }
        catch [System.Net.Http.HttpRequestException] { }
        Start-Sleep -Milliseconds 200
    }
    if ($null -eq $health -or $health.Status -ne 200) {
        throw "Server did not become ready. $(Get-ProcessLogs $serverProcess)"
    }
    Assert-Status '/health' $health 200

    $models = Send-ApiRequest 'GET' '/v1/models'
    Assert-Status '/v1/models' $models 200
    if ($null -eq $models.Json -or $models.Json.object -ne 'list' -or
        $models.Json.PSObject.Properties.Name -notcontains 'data') {
        throw "/v1/models did not return an OpenAI model list: $($models.Content)"
    }

    $state = Send-ApiRequest 'GET' '/internal/state'
    Assert-Status '/internal/state' $state 200
    if ($null -eq $state.Json -or $state.Json.is_loaded -ne $false) {
        throw "/internal/state should report an unloaded model: $($state.Content)"
    }

    $plainText = [System.Net.Http.StringContent]::new('{}', [System.Text.Encoding]::UTF8, 'text/plain')
    try {
        $unsupported = $client.PostAsync("$baseUri/v1/chat/completions", $plainText).GetAwaiter().GetResult()
        try { Assert-Status '/v1/chat/completions Content-Type' ([pscustomobject] @{ Status = [int] $unsupported.StatusCode; Content = $unsupported.Content.ReadAsStringAsync().GetAwaiter().GetResult() }) 415 }
        finally { $unsupported.Dispose() }
    }
    finally { $plainText.Dispose() }

    $validModel = Join-Path $logDir 'unused.gguf'
    New-Item -ItemType File -Path $validModel | Out-Null
    $validMessage = @(@{ role = 'user'; content = 'smoke test' })
    $badRequests = @(
        @{ Name = 'missing messages'; Parameter = 'messages'; Body = @{ model = $validModel } },
        @{ Name = 'model ../'; Parameter = 'model'; Traversal = $true; Body = @{ model = "$logDir/sub/../unused.gguf"; messages = $validMessage } },
        @{ Name = 'model ..\'; Parameter = 'model'; Traversal = $true; Body = @{ model = "$logDir\sub\..\unused.gguf"; messages = $validMessage } },
        @{ Name = 'lora ../'; Parameter = 'lora'; Traversal = $true; Body = @{ model = $validModel; lora = "$logDir/sub/../unused.bin"; messages = $validMessage } },
        @{ Name = 'lora ..\'; Parameter = 'lora'; Traversal = $true; Body = @{ model = $validModel; lora = "$logDir\sub\..\unused.bin"; messages = $validMessage } },
        @{ Name = 'top_p'; Parameter = 'top_p'; Body = @{ model = $validModel; messages = $validMessage; top_p = 1.1 } },
        @{ Name = 'top_k'; Parameter = 'top_k'; Body = @{ model = $validModel; messages = $validMessage; top_k = 0 } },
        @{ Name = 'temperature'; Parameter = 'temperature'; Body = @{ model = $validModel; messages = $validMessage; temperature = -1 } },
        @{ Name = 'max_tokens'; Parameter = 'max_tokens'; Body = @{ model = $validModel; messages = $validMessage; max_tokens = 0 } },
        @{ Name = 'max_completion_tokens'; Parameter = 'max_tokens'; Body = @{ model = $validModel; messages = $validMessage; max_completion_tokens = 0 } }
    )
    foreach ($case in $badRequests) {
        $response = Send-ApiRequest 'POST' '/v1/chat/completions' $case.Body
        Assert-Status "/v1/chat/completions $($case.Name)" $response 400
        if ($null -eq $response.Json -or
            $response.Json.PSObject.Properties.Name -notcontains 'error') {
            throw "Missing OpenAI error object for $($case.Name): $($response.Content)"
        }
        if ($response.Json.error.param -ne $case.Parameter) {
            throw "Unexpected error parameter for $($case.Name): $($response.Content)"
        }
        if ($case.Traversal -and $response.Json.error.message -notmatch '\.\.') {
            throw "Parent traversal was not the rejected condition for $($case.Name): $($response.Content)"
        }
    }

    $stateAfter = Send-ApiRequest 'GET' '/internal/state'
    Assert-Status '/internal/state after rejected requests' $stateAfter 200
    if ($stateAfter.Json.is_loaded -ne $false) {
        throw "A rejected request loaded the model: $($stateAfter.Content)"
    }
    Write-Host "Passed $passed local API smoke checks; no model generation was requested."
}
finally {
    if ($null -ne $serverProcess) { Stop-TestProcess $serverProcess.Process }
    $client.Dispose()
    foreach ($file in @(Get-ChildItem -LiteralPath $logDir -File -ErrorAction SilentlyContinue)) {
        Remove-Item -LiteralPath $file.FullName -Force
    }
    Remove-Item -LiteralPath $logDir -Force
}
