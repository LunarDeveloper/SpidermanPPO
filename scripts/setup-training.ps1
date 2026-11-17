$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    py -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    & ./.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & ./.venv/Scripts/python.exe -m pip install -r training/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
    & ./.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { throw 'GPU verification failed; CPU fallback is disabled.' }
} finally {
    Pop-Location
}
