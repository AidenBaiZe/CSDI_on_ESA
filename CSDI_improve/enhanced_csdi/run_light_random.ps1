$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StudyRoot = Split-Path -Parent (Split-Path -Parent $ProjectRoot)
$PythonExe = "D:\python\python.exe"
$BaselineRunner = Join-Path $StudyRoot "CSDI\ESA_mission1\run_mission1.py"
$BaselineConfig = Join-Path $ProjectRoot "light_random_baseline_config.yaml"
$EnhancedRunner = Join-Path $ProjectRoot "run_evaluation.py"
$EnhancedConfig = Join-Path $ProjectRoot "config.yaml"
$LogDir = Join-Path $ProjectRoot "results\mission1_enhanced_seed1\evaluation\light_random_logs"
$BaselineOutput = Join-Path $ProjectRoot "results\mission1_light_random_baseline_seed1\evaluation"
$SourceTen = Join-Path $StudyRoot "CSDI\ESA_mission1\results\mission1_random_seed1\evaluation\metrics_missing_10.json"
$BaselineTen = Join-Path $BaselineOutput "metrics_missing_10.json"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $BaselineOutput | Out-Null

if ((Test-Path -LiteralPath $SourceTen) -and -not (Test-Path -LiteralPath $BaselineTen)) {
    Copy-Item -LiteralPath $SourceTen -Destination $BaselineTen
}

& $PythonExe $BaselineRunner --mode evaluate --config $BaselineConfig *>> (Join-Path $LogDir "baseline.log")
if ($LASTEXITCODE -ne 0) { throw "Baseline light-random evaluation failed with exit code $LASTEXITCODE" }

& $PythonExe $EnhancedRunner light_random --config $EnhancedConfig *>> (Join-Path $LogDir "enhanced.log")
if ($LASTEXITCODE -ne 0) { throw "Enhanced light-random evaluation failed with exit code $LASTEXITCODE" }

@{
    status = "completed"
    completed_at = (Get-Date).ToString("o")
} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $LogDir "completed.json")
