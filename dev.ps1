<#
.SYNOPSIS
    TriSign tasks for Windows (Windows PowerShell 5.1 or PowerShell 7). Replaces the Makefile.

.DESCRIPTION
    .\dev.ps1 install   create .venv, install Python packages and the web app
    .\dev.ps1 models    download the MediaPipe models
    .\dev.ps1 demo      build the synthetic demo model and sign library
    .\dev.ps1 backend   API on http://localhost:8000   (keep this window open)
    .\dev.ps1 web       app on http://localhost:3000   (in a second window)
    .\dev.ps1 test      run all tests

    If Windows refuses to run the script, allow local scripts once:
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    or run a single task with:
        powershell -ExecutionPolicy Bypass -File .\dev.ps1 install
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet("install", "models", "demo", "backend", "web", "test", "help")]
    [string]$Task = "help",

    # Install PyTorch with CUDA (NVIDIA GPU) instead of the CPU build, e.g. -Cuda cu124
    [string]$Cuda = ""
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-Step {
    param([string]$Exe, [string[]]$Arguments, [string]$Dir = $Root)
    Push-Location $Dir
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Step failed (exit $LASTEXITCODE): $Exe $($Arguments -join ' ')" }
    }
    finally { Pop-Location }
}

function Assert-Venv {
    if (-not (Test-Path $Py)) { throw "No Python environment yet. Run: .\dev.ps1 install" }
}

function Find-BasePython {
    # Prefer Python 3.12 or 3.11 (best wheel support for PyTorch and MediaPipe), then whatever is on PATH.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.13")) {
            # A missing version writes to stderr; don't let that stop the script.
            $old = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            & py "-$v" -c "import sys" 2>&1 | Out-Null
            $ok = ($LASTEXITCODE -eq 0)
            $ErrorActionPreference = $old
            if ($ok) { return ,@("py", "-$v") }
        }
    }
    # The leading comma keeps a one-item array from collapsing into a plain string.
    if (Get-Command python -ErrorAction SilentlyContinue) { return ,@("python") }
    throw "Python not found. Install Python 3.12 from https://www.python.org/downloads/ (tick 'Add to PATH')."
}

switch ($Task) {
    "install" {
        if (-not (Test-Path $Py)) {
            $base = Find-BasePython
            $baseArgs = @($base | Select-Object -Skip 1)  # @() keeps it an array even with one item
            Write-Host "Creating .venv with $($base -join ' ')"
            Invoke-Step $base[0] ($baseArgs + @("-m", "venv", (Join-Path $Root ".venv")))
        }
        Invoke-Step $Py @("-m", "pip", "install", "--upgrade", "pip")
        if ($Cuda) {
            Invoke-Step $Py @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/$Cuda")
        }
        Invoke-Step $Py @("-m", "pip", "install", "-e", "ml[dev]", "-e", "backend[dev]")
        # MediaPipe for Python is only needed to extract landmarks from your own videos.
        try { Invoke-Step $Py @("-m", "pip", "install", "-e", "ml[extract]") }
        catch { Write-Host "MediaPipe for Python did not install (only needed for video extraction): $_" -ForegroundColor Yellow }
        if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
            throw "Node.js not found. Install the LTS version from https://nodejs.org/ and run install again."
        }
        Invoke-Step "npm.cmd" @("install") (Join-Path $Root "web")
        Write-Host "`nInstalled. Next: .\dev.ps1 models, then .\dev.ps1 demo"
    }

    "models" {
        $base = "https://storage.googleapis.com/mediapipe-models"
        $files = [ordered]@{
            "ml\models\mediapipe\pose_landmarker_full.task"   = "$base/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            "ml\models\mediapipe\hand_landmarker.task"        = "$base/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            "ml\models\mediapipe\face_landmarker.task"        = "$base/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            "web\public\mediapipe\holistic_landmarker.task"   = "$base/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
        }
        $ProgressPreference = "SilentlyContinue"  # the progress bar makes downloads far slower in PowerShell 5.1
        foreach ($target in $files.Keys) {
            $path = Join-Path $Root $target
            New-Item -ItemType Directory -Force -Path (Split-Path $path) | Out-Null
            Write-Host "Downloading $target"
            Invoke-WebRequest -Uri $files[$target] -OutFile $path -UseBasicParsing
        }
        Write-Host "MediaPipe models downloaded."
    }

    "demo" {
        Assert-Venv
        # SYNTHETIC demo: signs named SIGN00..SIGN11. Proves the pipeline; does not recognize real signs.
        Invoke-Step $Py @("-m", "trisign_ml.prepare", "synthetic", "--out", "data\synthetic\ase",
                          "--classes", "12", "--signers", "30", "--per-signer", "3")
        Invoke-Step $Py @("-m", "trisign_ml.train", "--signs", "ase=data\synthetic\ase", "--out", "runs\synthetic",
                          "--epochs", "12", "--batch", "32", "--dim", "96", "--layers", "2", "--workers", "0", "--lr", "3e-3")
        Invoke-Step $Py @("-m", "trisign_ml.export_onnx", "runs\synthetic\best.pt", "--out", "web\public\models")
        Invoke-Step $Py @("-m", "trisign_ml.build_sign_library", "--lang", "ase", "--data", "data\synthetic\ase",
                          "--out", "backend\data\sign_library\ase.json")
        Write-Host "`nDemo ready. Run .\dev.ps1 backend and, in a second window, .\dev.ps1 web"
        Write-Host "Then open http://localhost:3000 and type SIGN03 SIGN07 on the speaker side."
    }

    "backend" {
        Assert-Venv
        if (-not $env:OPENAI_API_KEY -and -not $env:ANTHROPIC_API_KEY) {
            Write-Host "Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set: using the rule-based fallback." -ForegroundColor Yellow
        }
        Invoke-Step $Py @("-m", "uvicorn", "app.main:app", "--reload", "--port", "8000") (Join-Path $Root "backend")
    }

    "web" {
        Invoke-Step "npm.cmd" @("run", "dev") (Join-Path $Root "web")
    }

    "test" {
        Assert-Venv
        Invoke-Step $Py @("-m", "pytest", "-q") (Join-Path $Root "ml")
        Invoke-Step $Py @("-m", "pytest", "-q") (Join-Path $Root "backend")
        Invoke-Step "npm.cmd" @("test") (Join-Path $Root "web")
        Write-Host "All tests passed."
    }

    default { Get-Help $PSCommandPath -Detailed }
}
