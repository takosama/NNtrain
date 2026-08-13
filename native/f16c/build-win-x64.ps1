[CmdletBinding()]
param(
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw 'Visual Studio Build Tools (vswhere.exe) was not found.'
}

$installation = & $vswhere -latest -products '*' `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath
if ([string]::IsNullOrWhiteSpace($installation)) {
    throw 'The Visual C++ x64 build tools are not installed.'
}

$vcvars = Join-Path $installation 'VC\Auxiliary\Build\vcvars64.bat'
$source = Join-Path $PSScriptRoot 'NNtrain.F16C.cpp'
$outputDirectory = Join-Path $repositoryRoot 'NNtrain.Core\runtimes\win-x64\native'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$output = Join-Path $outputDirectory 'NNtrain.F16C.dll'
$object = Join-Path $outputDirectory 'NNtrain.F16C.obj'

cmd.exe /d /s /c "`"$vcvars`" >nul && cl.exe /nologo /std:c++20 /O2 /fp:precise /LD /arch:AVX2 /Fo`"$object`" `"$source`" /Fe:`"$output`" /link /NOLOGO"
if ($LASTEXITCODE -ne 0) {
    throw "F16C native build failed with exit code $LASTEXITCODE."
}

Write-Host "Built $output"
