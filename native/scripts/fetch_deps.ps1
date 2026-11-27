# Fetches the TensorRT C++ headers and builds import libraries from the DLLs that
# ship with the Python wheels. Run once before configuring CMake.
param(
  [string]$PythonSite = (Join-Path $PSScriptRoot "..\..\.venv\Lib\site-packages"),
  [string]$Tag = "release/11.4"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$include = Join-Path $root "deps\include"
$lib = Join-Path $root "deps\lib"
New-Item -ItemType Directory -Force -Path $include, $lib | Out-Null

Write-Host "downloading TensorRT headers from $Tag"
$listing = Invoke-RestMethod "https://api.github.com/repos/NVIDIA/TensorRT/contents/include?ref=$Tag"
foreach ($file in $listing | Where-Object { $_.name -like "*.h" }) {
  Invoke-WebRequest -Uri $file.download_url -OutFile (Join-Path $include $file.name)
}
Write-Host ("headers: " + (Get-ChildItem $include -Filter *.h).Count)

Write-Host "downloading pocketfft and cpp-httplib"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mreineck/pocketfft/cpp/pocketfft_hdronly.h" `
  -OutFile (Join-Path $include "pocketfft_hdronly.h")
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h" `
  -OutFile (Join-Path $include "httplib.h")

$msvcRoots = @(
  "C:\Program Files\Microsoft Visual Studio",
  "C:\Program Files (x86)\Microsoft Visual Studio"
)

function Find-MsvcTool([string]$Name) {
  foreach ($root in $msvcRoots) {
    if (-not (Test-Path $root)) { continue }
    $hit = Get-ChildItem $root -Recurse -Filter $Name -ErrorAction SilentlyContinue |
      Where-Object { $_.FullName -like "*Hostx64\x64*" } |
      Select-Object -First 1
    if ($hit) { return $hit.FullName }
  }
  return $null
}

$dumpbin = Find-MsvcTool "dumpbin.exe"
$libexe = Find-MsvcTool "lib.exe"
if (-not $dumpbin -or -not $libexe) {
  throw "MSVC x64 tools not found; install the C++ build tools or run this from a Developer PowerShell"
}

# nvinfer: runtime, nvonnxparser: ONNX import for build-engine, plugin: built-in plugins.
foreach ($name in @("nvinfer_11", "nvonnxparser_11", "nvinfer_plugin_11")) {
  $dll = Join-Path $PythonSite "tensorrt_libs\$name.dll"
  if (-not (Test-Path $dll)) {
    throw "$name.dll not found in $PythonSite\tensorrt_libs; install tensorrt-cu12 first"
  }

  $names = & $dumpbin /exports $dll |
    Where-Object { $_ -match "^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+\S+$" } |
    ForEach-Object { ($_ -split "\s+")[-1] }

  $def = Join-Path $lib "$name.def"
  "LIBRARY $name`nEXPORTS" | Set-Content $def -Encoding ASCII
  $names | Add-Content $def -Encoding ASCII

  $out = Join-Path $lib "$name.lib"
  & $libexe /def:$def /machine:x64 /out:$out | Out-Null
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $out)) {
    throw "lib.exe could not build $out"
  }
  Write-Host ("import library: " + $out)
}
