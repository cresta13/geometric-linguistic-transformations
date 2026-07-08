$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Dist = Join-Path $Root "dist"
$OutRoot = Join-Path $Root "results\experiments\glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256_results"
$Stdout = Join-Path $Dist "glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256.out.log"
$Stderr = Join-Path $Dist "glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256.err.log"
$Monitor = Join-Path $Dist "glt_molt_spectral_pca_sweep_5m_160t_a100_300null_g256.monitor.log"

New-Item -ItemType Directory -Force -Path $Dist | Out-Null

function Write-Monitor($Message) {
    $line = "$(Get-Date -Format s) $Message"
    Add-Content -Path $Monitor -Value $line -Encoding UTF8
}

Write-Monitor "waiting for memory before compact PCA sweep"

while ($true) {
    wsl --shutdown 2>$null
    Start-Sleep -Seconds 5

    $os = Get-CimInstance Win32_OperatingSystem
    $freeMb = [math]::Round($os.FreePhysicalMemory / 1024, 0)
    Write-Monitor "free_mb=$freeMb"

    if ($freeMb -ge 6000) {
        break
    }

    Start-Sleep -Seconds 60
}

$env:GLT_MOLT_PCA_SWEEP_OUT_ROOT = $OutRoot
$env:GLT_MOLT_PCA_SWEEP_DIMS = "64,128,256"
$env:GLT_MOLT_PCA_SWEEP_ALPHAS = "100"
$env:GLT_MOLT_PCA_SWEEP_NULLS = "300"
$env:GLT_MOLT_PCA_SWEEP_GIVENS = "256"
$env:GLT_MOLT_PCA_SWEEP_TEMPLATES_PER_LANGUAGE = "160"
$env:GLT_MOLT_BATCH_SIZE = "1"
$env:GLT_MOLT_MODELS = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2,sentence-transformers/LaBSE,bert-base-multilingual-cased,xlm-roberta-base,distilbert-base-multilingual-cased"
$env:LIE_DEVICE = "cpu"
$env:LIE_TRUST_REMOTE_CODE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

Write-Monitor "starting compact PCA sweep"
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python (Join-Path $Root "scripts\run_glt_molt_spectral_pca_sweep.py") 1>> $Stdout 2>> $Stderr
$code = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
Write-Monitor "finished compact PCA sweep exit_code=$code"
exit $code
