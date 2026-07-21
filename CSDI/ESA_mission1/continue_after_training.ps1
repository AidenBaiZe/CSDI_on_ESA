$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$trainingDirectory = Join-Path $PSScriptRoot "results\mission1_random_seed1\training"
$finalCheckpoint = Join-Path $trainingDirectory "checkpoint_step_032000.pth"
$logDirectory = Join-Path $PSScriptRoot "results\mission1_random_seed1"
$statusPath = Join-Path $logDirectory "continuation_status.json"

function Write-Status([string]$status, [string]$message) {
    $payload = [ordered]@{
        status = $status
        message = $message
        updated_at = (Get-Date).ToString("o")
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-Status "waiting_for_training" "Waiting for checkpoint_step_032000.pth"
while (-not (Test-Path -LiteralPath $finalCheckpoint)) {
    $trainingProcess = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -like '*run_mission1.py --mode train*'
    }
    if (-not $trainingProcess) {
        Write-Status "failed" "Training stopped before the final checkpoint was created"
        throw "Training stopped before the final checkpoint was created"
    }
    Start-Sleep -Seconds 30
}

Write-Status "evaluating" "Running 10%, 50%, and 90% evaluations"
python run_mission1.py --mode evaluate
if ($LASTEXITCODE -ne 0) {
    Write-Status "failed" "Evaluation command failed with exit code $LASTEXITCODE"
    throw "Evaluation failed with exit code $LASTEXITCODE"
}

Write-Status "reporting" "Generating summary.json, CSV files, figures, and Chinese report"
python run_mission1.py --mode report
if ($LASTEXITCODE -ne 0) {
    Write-Status "failed" "Report command failed with exit code $LASTEXITCODE"
    throw "Report generation failed with exit code $LASTEXITCODE"
}

Write-Status "complete" "Training, all evaluations, and reporting are complete"
