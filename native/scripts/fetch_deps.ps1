# Fetches the TensorRT C++ headers and builds an import library from the DLL that
# ships with the Python wheels. Run once before configuring CMake.
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

Write-Host "downloading pocketfft"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mreineck/pocketfft/cpp/pocketfft_hdronly.h" `
  -OutFile (Join-Path $include "pocketfft_hdronly.h")

$dll = Join-Path $PythonSite "tensorrt_libs\nvinfer_11.dll"
if (-not (Test-Path $dll)) { throw "nvinfer_11.dll not found in $PythonSite" }

$dumpbin = (Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter dumpbin.exe |
  Select-Object -First 1).FullName
$libexe = (Get-ChildItem "C:\Program Files (x86)\Microsoft Visual Studio" -Recurse -Filter lib.exe |
  Where-Object { $_.FullName -notlike "*Hostx86*" } | Select-Object -First 1).FullName
if (-not $dumpbin -or -not $libexe) { throw "MSVC tools not found; run this from a Developer PowerShell" }

$names = & $dumpbin /exports $dll |
  Where-Object { $_ -match "^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+\S+$" } |
  ForEach-Object { ($_ -split "\s+")[-1] }
$def = Join-Path $lib "nvinfer_11.def"
"LIBRARY nvinfer_11`nEXPORTS" | Set-Content $def -Encoding ASCII
$names | Add-Content $def -Encoding ASCII

& $libexe /def:$def /machine:x64 /out:(Join-Path $lib "nvinfer_11.lib") | Out-Null
Write-Host ("import library: " + (Join-Path $lib "nvinfer_11.lib"))
