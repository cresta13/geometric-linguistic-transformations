$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$env:STEERING_OUT_DIR = if ($env:STEERING_OUT_DIR) { $env:STEERING_OUT_DIR } else { "results/experiments/gpt2_activation_steering_pilot_results" }
$env:STEERING_MODELS = if ($env:STEERING_MODELS) { $env:STEERING_MODELS } else { "distilgpt2,gpt2" }
$env:STEERING_CLASSES = if ($env:STEERING_CLASSES) { $env:STEERING_CLASSES } else { "question,negation,modality,tense_shift" }
$env:STEERING_GAINS = if ($env:STEERING_GAINS) { $env:STEERING_GAINS } else { "0.75,1.5,3.0" }
$env:STEERING_CONTROLS = if ($env:STEERING_CONTROLS) { $env:STEERING_CONTROLS } else { "none,target,wrong_class,random_norm,negative_target" }
$env:STEERING_MAX_TEST_SOURCES = if ($env:STEERING_MAX_TEST_SOURCES) { $env:STEERING_MAX_TEST_SOURCES } else { "24" }
$env:STEERING_MAX_NEW_TOKENS = if ($env:STEERING_MAX_NEW_TOKENS) { $env:STEERING_MAX_NEW_TOKENS } else { "28" }

$python = Join-Path $root ".venv/Scripts/python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

New-Item -ItemType Directory -Force -Path "dist" | Out-Null

$stdout = "dist/gpt2_activation_steering_pilot.out.log"
$stderr = "dist/gpt2_activation_steering_pilot.err.log"

Start-Process -FilePath $python `
  -ArgumentList @("scripts/run_gpt2_activation_steering_pilot.py") `
  -WorkingDirectory $root `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru
