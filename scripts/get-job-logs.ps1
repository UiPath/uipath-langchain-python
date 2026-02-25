param(
    [Parameter(Mandatory = $false)][string]$JobKey,
    [switch][alias("Staging")]$Stg,
    [switch][alias("Alpha")]$Alp,
    [switch][alias("Prod")]$Prd,
    [string]$Ago = "24h",
    [switch]$Help
)

if ($Help -or -not $JobKey) {
    Write-Host @"
Usage: get-job-logs.ps1 -JobKey <key> [-Stg|-Alp|-Prd] [-Ago <offset>]

  -JobKey   Job key (required)
  -Stg      Staging environment
  -Alp      Alpha environment
  -Prd      Production environment (default)
  -Ago      How far back to search, e.g. 24h, 2d, 1d12h  (default: 24h)

Examples:
  .\get-job-logs.ps1 -JobKey abc123
  .\get-job-logs.ps1 -JobKey abc123 -Alp -Ago 48h
  .\get-job-logs.ps1 -JobKey abc123 -Stg -Ago 1d12h
"@
    exit 0
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Environment resolution ---
if (($Stg.IsPresent + $Alp.IsPresent + $Prd.IsPresent) -gt 1) {
    Write-Error "Specify at most one environment switch: -Stg, -Alp, or -Prd."
    exit 1
}
if ($Stg) { $EnvName = "stg" }
elseif ($Alp) { $EnvName = "alp" }
else { $EnvName = "prd" }

# Convert ##d##h Ago format to ISO 8601 duration for az monitor log-analytics query, e.g. "1d12h" -> "P1DT12H"
$LogAnalyticsTimespan = "P" + ($Ago.ToUpper() -replace '(\d+H)', 'T$1')

function Assert-LastExitCode {
    param([string]$Context)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "az command failed during: $Context (exit code $LASTEXITCODE)"
        exit 1
    }
}

function ConvertTo-ZlibBase64 {
    param([string]$Text)
    $inputBytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
    $ms = New-Object System.IO.MemoryStream
    $zlib = New-Object System.IO.Compression.ZLibStream($ms, [System.IO.Compression.CompressionMode]::Compress)
    $zlib.Write($inputBytes, 0, $inputBytes.Length)
    $zlib.Dispose()
    return [System.Convert]::ToBase64String($ms.ToArray())
}

# Check required extensions
$extensions = az extension list --query "[].name" -o json 2>$null | ConvertFrom-Json
foreach ($ext in @("resource-graph", "application-insights", "log-analytics")) {
    if ($extensions -notcontains $ext) {
        Write-Error @"
The '$ext' az extension is required but not installed.
Install it with:
    az extension add --name $ext
"@
        exit 1
    }
}

# Locate Serverless AppInsights via Resource Graph
$serverlessName = "srvless-$EnvName-global-appins"
Write-Host "Searching for Serverless AppInsights resource '$serverlessName'..."

$resourceSearchResultJson = az graph query -q @"
resources
| where type == 'microsoft.insights/components'
| where name == '$serverlessName'
| project id, subscriptionId, resourceGroup
| limit 1
"@ --query "data[0]" -o json
Assert-LastExitCode "Resource Graph query for AppInsights"

$resourceSearchResult = $resourceSearchResultJson | ConvertFrom-Json
if (-not $resourceSearchResult -or -not $resourceSearchResult.id) {
    Write-Error "Serverless AppInsights resource '$serverlessName' not found. Check that the environment name is correct."
    exit 1
}

$serverlessAppInsId = $resourceSearchResult.id

# Query Serverless AppInsights for RuntimeIdentifier + ExecutionInstanceId
Write-Host "Querying AppInsights for job key '$JobKey'..."

$serverlessQuery = @"
customEvents
| where name == 'Serverless.JobScheduled'
| where customDimensions['JobId'] == '$JobKey'
| extend RuntimeIdentifier = tostring(customDimensions['RuntimeIdentifier']),
         ExecutionInstanceId = tostring(customDimensions['ExecutionInstanceId'])
| project RuntimeIdentifier, ExecutionInstanceId
| limit 1
"@

$serverlessQueryResultJson = az monitor app-insights query `
    --ids $serverlessAppInsId `
    --analytics-query $serverlessQuery `
    --offset $Ago `
    -o json
Assert-LastExitCode "Serverless AppInsights query for JobScheduled event"

$serverlessQueryResult = $serverlessQueryResultJson | ConvertFrom-Json
$serverlessQueryResultRow = if ($serverlessQueryResult.tables -and $serverlessQueryResult.tables[0].rows -and $serverlessQueryResult.tables[0].rows.Count -gt 0) {
    $serverlessQueryResult.tables[0].rows[0]
}
else { $null }
if (-not $serverlessQueryResultRow -or $serverlessQueryResultRow.Count -lt 2) {
    Write-Error "No 'Serverless.JobScheduled' event found for JobKey '$JobKey' in '$serverlessName'. Verify the job key, environment, and time offset."
    exit 1
}

$RuntimeIdentifier = $serverlessQueryResultRow[0]
$ExecutionInstanceId = $serverlessQueryResultRow[1]

Write-Host "RuntimeId    : $RuntimeIdentifier"
Write-Host "ExecutionId  : $ExecutionInstanceId"

# Find Resource Group via Resource Graph
Write-Host "`nSearching for resource group containing '$RuntimeIdentifier'..."

$rgJson = az graph query -q @"
resourcecontainers
| where type == 'microsoft.resources/subscriptions/resourcegroups'
| where name contains '$RuntimeIdentifier'
| project name, subscriptionId
| limit 1
"@ --query "data[0]" -o json
Assert-LastExitCode "Resource Graph query for resource group"

$rgData = $rgJson | ConvertFrom-Json
if (-not $rgData -or -not $rgData.name) {
    Write-Error "No resource group found containing '$RuntimeIdentifier'. The runtime container may have been cleaned up."
    exit 1
}

$TargetRG = $rgData.name
$TargetSubscription = $rgData.subscriptionId

Write-Host "Resource Group : $TargetRG"

# Find Log Analytics Workspace via Resource Graph
Write-Host "Searching for Log Analytics workspace in '$TargetRG'..."

$workspaceJson = az graph query -q @"
resources
| where type == 'microsoft.operationalinsights/workspaces'
| where resourceGroup == '$TargetRG'
| where subscriptionId == '$TargetSubscription'
| project id, properties.customerId
| limit 1
"@ --query "data[0]" -o json
Assert-LastExitCode "Resource Graph query for Log Analytics workspace"

$workspaceData = $workspaceJson | ConvertFrom-Json
if (-not $workspaceData -or -not $workspaceData.id) {
    Write-Error "No Log Analytics workspace found in resource group '$TargetRG' (subscription: $TargetSubscription)."
    exit 1
}

$WorkspaceResourceId = $workspaceData.id
$WorkspaceGuid = $workspaceData.properties_customerId
$WorkspaceName = $WorkspaceResourceId.Split("/")[-1]

Write-Host "Log Analytics Workspace : $WorkspaceName"

# Query Log Analytics for Container Logs
Write-Host "`nSetting subscription context for Log Analytics query..."
az account set --subscription $TargetSubscription | Out-Null
Assert-LastExitCode "az account set for Log Analytics subscription"

# Query marker lines only
$markerQuery = @"
let containerId = toscalar(
    ContainerInventory
    | where ContainerHostname == '$ExecutionInstanceId'
    | where isnotempty(ContainerID)
    | summarize any(ContainerID)
);
ContainerLog
| where isnotempty(containerId)
| where ContainerID == containerId
| where LogEntry contains "==== Job $JobKey"
    or LogEntry contains "==== STATS JobKey: $JobKey"
    or LogEntry contains "[Python Server] Starting job $JobKey"
| order by TimeGenerated asc
| project TimeGenerated, LogEntry
"@

Write-Host "Querying marker lines for job '$JobKey' in container '$ExecutionInstanceId'..."
$markersJson = az monitor log-analytics query `
    --workspace $WorkspaceGuid `
    --analytics-query $markerQuery `
    --timespan $LogAnalyticsTimespan
Assert-LastExitCode "Log Analytics marker query"
$markerRows = if ($markersJson) { $markersJson | ConvertFrom-Json } else { @() }

# Parse run intervals from marker rows
$runs = [System.Collections.Generic.List[hashtable]]::new()
$runStart = $null
$isResume = $false

foreach ($row in $markerRows) {
    $entry = $row.LogEntry
    if ($null -eq $runStart -and $entry -like "*==== Job $JobKey*") {
        $runStart = $row.TimeGenerated
        $isResume = $false
    }
    elseif ($null -ne $runStart -and $entry -like "*`[Python Server`] Starting job $JobKey*") {
        $isResume = $entry -like "*--resume*"
    }
    elseif ($null -ne $runStart -and $entry -like "*==== STATS JobKey: $JobKey*") {
        $runs.Add(@{ Start = $runStart; End = $row.TimeGenerated; IsResume = $isResume })
        $runStart = $null
        $isResume = $false
    }
}
# Open run: job still executing (no STATS yet)
if ($null -ne $runStart -and $markerRows.Count -gt 0) {
    $runs.Add(@{ Start = $runStart; End = $markerRows[-1].TimeGenerated; IsResume = $isResume })
}

if ($runs.Count -eq 0) {
    Write-Error "No '==== Job $JobKey' markers found in container logs. Job may not have run in this container within the $Ago window."
    exit 1
}

# Query full logs for detected intervals
$runSubqueries = @(for ($i = 0; $i -lt $runs.Count; $i++) {
        $endPlusOne = ([datetime]$runs[$i].End).AddSeconds(1).ToString("yyyy-MM-dd HH:mm:ss")
        "(ContainerLog | where isnotempty(containerId) | where ContainerID == containerId `
        | where TimeGenerated between (datetime('$($runs[$i].Start)') .. datetime('$endPlusOne')) `
        | extend RunIndex = $i)"
    })
$unionBody = $runSubqueries -join ",`n"

$logQuery = @"
let containerId = toscalar(
    ContainerInventory
    | where ContainerHostname == '$ExecutionInstanceId'
    | where isnotempty(ContainerID)
    | summarize any(ContainerID)
);
union
$unionBody
| order by TimeGenerated asc
| project TimeGenerated, RunIndex, LogEntry, Computer, Image
"@

Write-Host "Found $($runs.Count) run(s). Fetching logs for job-specific intervals..."
$logsJson = az monitor log-analytics query `
    --workspace $WorkspaceGuid `
    --analytics-query $logQuery `
    --timespan $LogAnalyticsTimespan
Assert-LastExitCode "Log Analytics job-filtered query"

$logRows = if ($logsJson) { $logsJson | ConvertFrom-Json } else { @() }
$logCount = if ($logRows) { @($logRows).Count } else { 0 }

# Generate Azure Portal Deep Link
$tenantId = az account show --query tenantId -o tsv
Assert-LastExitCode "Get tenant ID"

$scopeJson = '{"resources":[{"resourceId":"' + $WorkspaceResourceId + '"}]}'
$encodedScope = [System.Uri]::EscapeDataString($scopeJson)
$encodedQuery = [System.Uri]::EscapeDataString((ConvertTo-ZlibBase64 $logQuery.Trim()))

$portalEnd = [System.DateTime]::UtcNow
$portalStart = $portalEnd.Subtract([System.Xml.XmlConvert]::ToTimeSpan($LogAnalyticsTimespan))
$encodedTimespan = [System.Uri]::EscapeDataString(
    $portalStart.ToString("yyyy-MM-ddTHH:mm:ss.0000000Z") + "/" +
    $portalEnd.ToString("yyyy-MM-ddTHH:mm:ss.0000000Z"))

$portalLink = "https://portal.azure.com/#@$tenantId/blade/Microsoft_Azure_Monitoring_Logs/LogsBlade/source/LogsBlade.AnalyticsShareLinkToQuery/scope/$encodedScope/q/$encodedQuery/prettify/1/timespan/$encodedTimespan"

# Output Summary
Write-Host "`n--- SUMMARY ---"
Write-Host ("Environment    : " + $EnvName)
Write-Host ("Job Key        : " + $JobKey)
Write-Host ("RuntimeId      : " + $RuntimeIdentifier)
Write-Host ("ExecutionId    : " + $ExecutionInstanceId)
Write-Host ("Resource Group : " + $TargetRG)
Write-Host ("Workspace      : " + $WorkspaceName)
Write-Host ("Log lines      : " + $logCount)

if ($logCount -eq 0) {
    Write-Error "No log entries found for the detected run intervals."
    exit 1
}

Write-Host "`nPortal link:"
Write-Host $portalLink
