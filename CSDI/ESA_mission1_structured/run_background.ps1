$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $here
$result = Join-Path $here "results\mission1_structured_only_seed1"
$statusPath = Join-Path $result "status.json"
$pipelineLog = Join-Path $result "pipeline.log"
New-Item -ItemType Directory -Force -Path $result | Out-Null

function Write-WaitStatus([string]$message) {
    $payload = [ordered]@{
        status = "waiting_for_gpu"
        stage = "waiting_for_gpu"
        message = $message
        updated_at = [DateTimeOffset]::Now.ToString("o")
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

while (Get-Process -Name "TheBazaar" -ErrorAction SilentlyContinue) {
    Write-WaitStatus "TheBazaar is using the GPU; the structured-only experiment will start automatically after it exits"
    Start-Sleep -Seconds 30
}

$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $root
python "$here\run_structured.py" --mode full --resume 2>&1 | Tee-Object -FilePath $pipelineLog -Append
exit $LASTEXITCODE
