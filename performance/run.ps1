$requiredVars = @(
    "PERF_USERNAME",
    "PERF_PASSWORD",
    "MODE",
    "INPUT_ARGUMENTS",
    "FOLDER_ID",
    "DURATION",
    "RATE",
    "TIME_UNIT",
    "PRE_ALLOCATED_VUS",
    "MAX_VUS",
    "CLOUD_ENVIRONMENT",
    "ORGANIZATION_NAME",
    "ORGANIZATION_ID",
    "TENANT_NAME",
    "TENANT_ID",
    "HALT_WAIT_THRESHOLD")
foreach ($var in $requiredVars) {
    if (-not (Get-Item "env:$var" -ErrorAction SilentlyContinue)) {
        Write-Error "$var environment variable is required"
        exit 1
    }
}

function MoveResultFiles($targetDir) {
    $filesToMove = @("summary.json", "k6.log")
    foreach ($file in $filesToMove) {
        if (Test-Path -Path $file) {
            Move-Item -Path $file -Destination "$targetDir\" -Force
        }
    }
}

$start_timestamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
Write-Output "Started at $start_timestamp"

$OUTPUT_DIR = "results\$start_timestamp"

npm install
npm run build

$START_OUTPUT_DIR = "${OUTPUT_DIR}_start"
New-Item -ItemType Directory -Path $START_OUTPUT_DIR -Force | Out-Null

k6 run `
  --out "json=$START_OUTPUT_DIR\raw_results.json" `
  -e "PERF_USERNAME=$($env:PERF_USERNAME)" `
  -e "PERF_PASSWORD=$($env:PERF_PASSWORD)" `
  -e "EXEC=$($env:MODE)" `
  -e "PROCESS_KEY=$($env:PROCESS_KEY)" `
  -e "SOLUTION_PROJECT_ID=$($env:SOLUTION_PROJECT_ID)" `
  -e "INPUT_ARGUMENTS=$($env:INPUT_ARGUMENTS)" `
  -e "FOLDER_ID=$($env:FOLDER_ID)" `
  -e "DURATION=$($env:DURATION)" `
  -e "RATE=$($env:RATE)" `
  -e "TIME_UNIT=$($env:TIME_UNIT)" `
  -e "PRE_ALLOCATED_VUS=$($env:PRE_ALLOCATED_VUS)" `
  -e "MAX_VUS=$($env:MAX_VUS)" `
  -e "CLOUD_ENVIRONMENT=$($env:CLOUD_ENVIRONMENT)" `
  -e "ORGANIZATION_NAME=$($env:ORGANIZATION_NAME)" `
  -e "ORGANIZATION_ID=$($env:ORGANIZATION_ID)" `
  -e "TENANT_NAME=$($env:TENANT_NAME)" `
  -e "TENANT_ID=$($env:TENANT_ID)" `
  -e "HALT_WAIT_THRESHOLD=$($env:HALT_WAIT_THRESHOLD)" `
  -e "SOCKET_TIMEOUT_MINUTES=$($env:SOCKET_TIMEOUT_MINUTES)" `
  --log-output=file=.\k6.log `
  --log-format json `
  ./dist/main.js

MoveResultFiles $START_OUTPUT_DIR

$end_timestamp = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
Write-Output "Finished at $end_timestamp"
