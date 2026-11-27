# Stages the prebuilt engine builder with the TensorRT runtime it needs and zips it.
param(
  [Parameter(Mandatory = $true)][string]$Site,
  [string]$Version = "",
  [string]$Arch = "sm89",
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
if (-not $OutDir) { $OutDir = Join-Path $root "dist" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$exe = Get-ChildItem (Join-Path $root "native") -Recurse -Filter "build-engine.exe" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $exe) {
  throw "build-engine.exe not found; build the tools target first"
}

$stage = Join-Path $OutDir "qwen3-asr-build-engine-$Arch"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Copy-Item $exe.FullName $stage

$runtime = Get-ChildItem (Join-Path $root "native") -Recurse -Filter "qwen3-asr-native.exe" -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $runtime) {
  throw "qwen3-asr-native.exe not found; build the runtime target first"
}
Copy-Item $runtime.FullName $stage

$sndfile = Join-Path $Site "_soundfile_data\libsndfile_x64.dll"
if (-not (Test-Path $sndfile)) { throw "libsndfile_x64.dll not found; pip install soundfile" }
Copy-Item $sndfile $stage

$assetDir = Join-Path $stage "assets"
New-Item -ItemType Directory -Force -Path $assetDir | Out-Null
foreach ($asset in @("mel.bin", "tokenizer.bin")) {
  $path = Join-Path $root "native\assets\$asset"
  if (-not (Test-Path $path)) { throw "$asset missing; run native/tools/gen_assets.py" }
  Copy-Item $path $assetDir
}
foreach ($dll in @("nvinfer_11.dll", "nvonnxparser_11.dll", "nvinfer_plugin_11.dll",
    "nvinfer_builder_resource_$($Arch)_11.dll")) {
  $path = Join-Path $Site "tensorrt_libs\$dll"
  if (-not (Test-Path $path)) { throw "$dll not found in $Site\tensorrt_libs" }
  Copy-Item $path $stage
}

$cudart = @(
  (Join-Path $Site "nvidia\cuda_runtime\bin\cudart64_12.dll"),
  (Join-Path $Site "torch\lib\cudart64_12.dll")
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $cudart) { throw "cudart64_12.dll not found; install nvidia-cuda-runtime-cu12" }
Copy-Item $cudart $stage

foreach ($pattern in @("tensorrt_cu12_libs-*\LICENSE.txt", "nvidia_cuda_runtime_cu12-*\LICENSE.txt")) {
  Get-ChildItem (Join-Path $Site $pattern) -ErrorAction SilentlyContinue | ForEach-Object {
    $name = $_.Directory.Name -replace "-[\d.]+\..*$", ""
    Copy-Item $_.FullName (Join-Path $stage "LICENSE-$name.txt")
  }
}

& (Join-Path $stage "build-engine.exe") --version | Set-Content (Join-Path $stage "VERSION.txt") -Encoding ASCII

@"
Build TensorRT engines from the ONNX graphs published by this project and run
them, without Python or PyTorch. Needs an NVIDIA driver; the TensorRT and CUDA
runtime DLLs are bundled.

This package is for $Arch GPUs. build-engine loads only the builder resource
that matches the installed card. Compute capabilities: sm75 Turing (T4, RTX 20),
sm80 A100, sm86 RTX 30, sm89 RTX 40, sm90 H100, sm100 Blackwell, sm120 RTX 50.
qwen3-asr-native.exe is a fat binary and runs on all of those.

  build-engine.exe --onnx encoder-window.bf16.onnx --out encoder-window.bf16.engine
  build-engine.exe --onnx asr-prefill.onnx --out asr-prefill.fp16.engine --max-seq 512

  qwen3-asr-native.exe --model decoder-int4-g32 --encoder encoder-window.bf16.engine ^
    --decoder asr-prefill.fp16.engine --assets assets --language English audio.flac

A sibling <file>.onnx.data must stay next to the .onnx. Each build also writes
<engine>.json with the TensorRT version, CUDA architecture, and profile shapes.
Forced alignment also needs the aligner assets, published separately as
qwen3-asr-aligner-assets.zip; unpack them into the same assets directory.
Run build-engine.exe --help for every option.
"@ | Set-Content (Join-Path $stage "HOWTO.txt") -Encoding ASCII

$suffix = if ($Version) { "-$Version" } else { "" }
$zip = Join-Path $OutDir "qwen3-asr-build-engine$suffix-$Arch.zip"
Remove-Item $zip -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip -CompressionLevel Fastest

$hash = (Get-FileHash $zip -Algorithm SHA256).Hash
"$hash  $(Split-Path $zip -Leaf)" | Set-Content "$zip.sha256" -Encoding ASCII

$size = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "wrote $zip ($size MB)"
