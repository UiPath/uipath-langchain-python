param(
    [Parameter(Mandatory = $false)][string]$JobKey,
    [switch][alias("Staging")]$Stg,
    [switch][alias("Alpha")]$Alp,
    [switch][alias("Prod")]$Prd,
    [string]$Ago = "24h",
    [int]$MaxLines = 5000,
    [switch]$NoAgentTelemetry,
    [switch]$Help
)

if ($Help -or -not $JobKey) {
    Write-Host @"
Usage: get-agent-logs-inline.ps1 -JobKey <key> [-Stg|-Alp|-Prd] [-Ago <offset>] [-MaxLines <n>] [-NoAgentTelemetry]

Returns job logs as a JSON object on stdout for programmatic consumption.
Progress messages go to the host (stderr-equivalent) and won't pollute captured output.

  -JobKey             Job key (required)
  -Stg                Staging environment
  -Alp                Alpha environment
  -Prd                Production environment (default)
  -Ago                How far back to search, e.g. 24h, 2d, 1d12h  (default: 24h)
  -MaxLines           Maximum log lines to return (default: 5000)
  -NoAgentTelemetry   Skip querying the agents AppInsights for Python-side telemetry

Output JSON schema:
  {
    "jobKey": "...",
    "environment": "prd|alp|stg",
    "runtimeIdentifier": "...",
    "executionInstanceId": "...",
    "resourceGroup": "...",
    "workspace": "...",
    "runs": [ { "index": 0, "start": "...", "end": "...", "isResume": false } ],
    "logCount": 123,
    "truncated": false,
    "timeline": [
      { "timestamp": "...", "source": "container", "runIndex": 0, "entry": "..." },
      { "timestamp": "...", "source": "telemetry", "type": "exception", "name": "...", "message": "..." }
    ],
    "agentTelemetry": { "operationIds": ["..."], "eventCount": 26 }
  }

Examples:
  `$json = pwsh ./get-agent-logs-inline.ps1 -JobKey abc123 | ConvertFrom-Json
  `$logs = (pwsh ./get-agent-logs-inline.ps1 -JobKey abc123 -Alp -Ago 48h | ConvertFrom-Json).logs
  `$tel  = (pwsh ./get-agent-logs-inline.ps1 -JobKey abc123 -Stg | ConvertFrom-Json).agentTelemetry
"@
    exit 0
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- Helpers ---

function Write-Status {
    param([string]$Message)
    [Console]::Error.WriteLine($Message)
}

function Write-JsonError {
    param([string]$Message)
    $errObj = @{ error = $Message } | ConvertTo-Json -Compress
    [Console]::Out.WriteLine($errObj)
    [Environment]::Exit(1)
}

function Assert-LastExitCode {
    param([string]$Context)
    if ($LASTEXITCODE -ne 0) {
        Write-JsonError "az command failed during: $Context (exit code $LASTEXITCODE)"
    }
}

# --- Environment resolution ---
if (($Stg.IsPresent + $Alp.IsPresent + $Prd.IsPresent) -gt 1) {
    Write-JsonError "Specify at most one environment switch: -Stg, -Alp, or -Prd."
}
if ($Stg) { $EnvName = "stg" }
elseif ($Alp) { $EnvName = "alp" }
else { $EnvName = "prd" }

# Convert ##d##h Ago format to ISO 8601 duration for az monitor log-analytics query
$LogAnalyticsTimespan = "P" + ($Ago.ToUpper() -replace '(\d+H)', 'T$1')

# Check required extensions
$extensions = az extension list --query "[].name" -o json 2>$null | ConvertFrom-Json
foreach ($ext in @("resource-graph", "application-insights", "log-analytics")) {
    if ($extensions -notcontains $ext) {
        Write-JsonError "The '$ext' az extension is required but not installed. Install it with: az extension add --name $ext"
    }
}

# --- Locate Serverless AppInsights via Resource Graph ---
$serverlessName = "srvless-$EnvName-global-appins"
Write-Status "Searching for Serverless AppInsights resource '$serverlessName'..."

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
    Write-JsonError "Serverless AppInsights resource '$serverlessName' not found. Check that the environment name is correct."
}

$serverlessAppInsId = $resourceSearchResult.id

# --- Query Serverless AppInsights for RuntimeIdentifier + ExecutionInstanceId ---
Write-Status "Querying AppInsights for job key '$JobKey'..."

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
    Write-JsonError "No 'Serverless.JobScheduled' event found for JobKey '$JobKey' in '$serverlessName'. Verify the job key, environment, and time offset."
}

$RuntimeIdentifier = $serverlessQueryResultRow[0]
$ExecutionInstanceId = $serverlessQueryResultRow[1]

Write-Status "RuntimeId    : $RuntimeIdentifier"
Write-Status "ExecutionId  : $ExecutionInstanceId"

# --- Find Resource Group via Resource Graph ---
Write-Status "Searching for resource group containing '$RuntimeIdentifier'..."

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
    Write-JsonError "No resource group found containing '$RuntimeIdentifier'. The runtime container may have been cleaned up."
}

$TargetRG = $rgData.name
$TargetSubscription = $rgData.subscriptionId

Write-Status "Resource Group : $TargetRG"

# --- Find Log Analytics Workspace via Resource Graph ---
Write-Status "Searching for Log Analytics workspace in '$TargetRG'..."

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
    Write-JsonError "No Log Analytics workspace found in resource group '$TargetRG' (subscription: $TargetSubscription)."
}

$WorkspaceResourceId = $workspaceData.id
$WorkspaceGuid = $workspaceData.properties_customerId
$WorkspaceName = $WorkspaceResourceId.Split("/")[-1]

Write-Status "Log Analytics Workspace : $WorkspaceName"

# --- Query Log Analytics for Container Logs ---
Write-Status "Setting subscription context for Log Analytics query..."
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

Write-Status "Querying marker lines for job '$JobKey' in container '$ExecutionInstanceId'..."
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
    Write-JsonError "No '==== Job $JobKey' markers found in container logs. Job may not have run in this container within the $Ago window."
}

# --- Query full logs for detected intervals ---
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
| project TimeGenerated, RunIndex, LogEntry
| limit $MaxLines
"@

Write-Status "Found $($runs.Count) run(s). Fetching logs for job-specific intervals..."
$logsJson = az monitor log-analytics query `
    --workspace $WorkspaceGuid `
    --analytics-query $logQuery `
    --timespan $LogAnalyticsTimespan
Assert-LastExitCode "Log Analytics job-filtered query"

$logRows = if ($logsJson) { $logsJson | ConvertFrom-Json } else { @() }
$logCount = if ($logRows) { @($logRows).Count } else { 0 }
$truncated = $logCount -ge $MaxLines

Write-Status "Retrieved $logCount log line(s)$(if ($truncated) { ' (truncated)' })."

# --- Query Agents AppInsights for Python-side telemetry ---
$agentTelemetryOutput = $null
if (-not $NoAgentTelemetry) {
    $agentsAppInsName = "agents-$EnvName-appins-ne-appins"
    Write-Status "`nSearching for Agents AppInsights resource '$agentsAppInsName'..."

    $agentsResourceJson = az graph query -q @"
resources
| where type == 'microsoft.insights/components'
| where name == '$agentsAppInsName'
| project id, subscriptionId, resourceGroup
| limit 1
"@ --query "data[0]" -o json
    Assert-LastExitCode "Resource Graph query for Agents AppInsights"

    $agentsResource = $agentsResourceJson | ConvertFrom-Json
    if (-not $agentsResource -or -not $agentsResource.id) {
        Write-Status "WARNING: Agents AppInsights resource '$agentsAppInsName' not found. Skipping agent telemetry."
    }
    else {
        $agentsAppInsId = $agentsResource.id

        # Step 1: Find operation_Id(s) via AgentRun events correlated by JobKey
        Write-Status "Querying Agents AppInsights for operation IDs (JobKey='$JobKey')..."

        $opIdQuery = @"
customEvents
| where customDimensions['JobKey'] == '$JobKey'
| where name in ('AgentRun.Start', 'AgentRun.Failed', 'AgentRun.Completed')
| summarize by operation_Id
"@

        $opIdResultJson = az monitor app-insights query `
            --ids $agentsAppInsId `
            --analytics-query $opIdQuery `
            --offset $Ago `
            -o json
        Assert-LastExitCode "Agents AppInsights query for operation IDs"

        $opIdResult = $opIdResultJson | ConvertFrom-Json
        $opIds = @()
        if ($opIdResult.tables -and $opIdResult.tables[0].rows) {
            $opIds = @($opIdResult.tables[0].rows | ForEach-Object { $_[0] } | Where-Object { $_ })
        }

        if ($opIds.Count -eq 0) {
            Write-Status "WARNING: No AgentRun events found for JobKey '$JobKey' in '$agentsAppInsName'. Skipping telemetry."
        }
        else {
            Write-Status "Found $($opIds.Count) operation ID(s): $($opIds -join ', ')"

            # Step 2: Query all telemetry tables for those operation_Id(s)
            $opIdFilter = ($opIds | ForEach-Object { "'$_'" }) -join ", "

            $telemetryQuery = @"
let opIds = dynamic([$opIdFilter]);
union
(customEvents     | where operation_Id in (opIds) | extend telemetryType = 'customEvent',  detailName = name, detailMessage = tostring(customDimensions)),
(requests         | where operation_Id in (opIds) | extend telemetryType = 'request',      detailName = name, detailMessage = strcat(resultCode, ' ', url)),
(dependencies     | where operation_Id in (opIds) | extend telemetryType = 'dependency',   detailName = name, detailMessage = strcat(resultCode, ' ', type, ' ', target, ' ', data)),
(traces           | where operation_Id in (opIds) | extend telemetryType = 'trace',        detailName = tostring(severityLevel), detailMessage = message),
(exceptions       | where operation_Id in (opIds) | extend telemetryType = 'exception',    detailName = type, detailMessage = strcat(outerMessage, ' | ', innermostMessage))
| order by timestamp asc
| project timestamp, telemetryType, operation_Id, detailName, detailMessage, customDimensions
| limit $MaxLines
"@

            Write-Status "Fetching correlated telemetry across all tables..."
            $telemetryJson = az monitor app-insights query `
                --ids $agentsAppInsId `
                --analytics-query $telemetryQuery `
                --offset $Ago `
                -o json
            Assert-LastExitCode "Agents AppInsights correlated telemetry query"

            $telemetryResult = $telemetryJson | ConvertFrom-Json
            $telemetryRows = @()
            if ($telemetryResult.tables -and $telemetryResult.tables[0].rows) {
                $cols = $telemetryResult.tables[0].columns | ForEach-Object { $_.name }
                $telemetryRows = @($telemetryResult.tables[0].rows | ForEach-Object {
                    $row = $_
                    $obj = [ordered]@{}
                    for ($c = 0; $c -lt $cols.Count; $c++) {
                        $obj[$cols[$c]] = $row[$c]
                    }
                    $obj
                })
            }

            Write-Status "Retrieved $($telemetryRows.Count) telemetry event(s)."

            $telemetryEvents = @($telemetryRows | ForEach-Object {
                $dims = $null
                if ($_['customDimensions']) {
                    try { $dims = $_['customDimensions'] | ConvertFrom-Json -AsHashtable } catch { $dims = $_['customDimensions'] }
                }
                [ordered]@{
                    timestamp       = $_['timestamp']
                    type            = $_['telemetryType']
                    operationId     = $_['operation_Id']
                    name            = $_['detailName']
                    message         = $_['detailMessage']
                    customDimensions = $dims
                }
            })

            $agentTelemetryOutput = [ordered]@{
                operationIds = $opIds
                eventCount   = $telemetryEvents.Count
                events       = $telemetryEvents
            }
        }
    }
}

# --- Build structured output ---
$runsOutput = @(for ($i = 0; $i -lt $runs.Count; $i++) {
    [ordered]@{
        index    = $i
        start    = $runs[$i].Start
        end      = $runs[$i].End
        isResume = $runs[$i].IsResume
    }
})

$logsOutput = @(foreach ($row in @($logRows)) {
    [ordered]@{
        timestamp = $row.TimeGenerated
        runIndex  = [int]$row.RunIndex
        entry     = $row.LogEntry
    }
})

# --- Build merged timeline (container logs + agent telemetry, sorted by timestamp) ---
$timelineEntries = [System.Collections.Generic.List[object]]::new()

foreach ($row in @($logRows)) {
    $timelineEntries.Add([ordered]@{
        timestamp = $row.TimeGenerated
        source    = "container"
        runIndex  = [int]$row.RunIndex
        entry     = $row.LogEntry
    })
}

if ($agentTelemetryOutput) {
    foreach ($evt in $agentTelemetryOutput.events) {
        $timelineEntries.Add([ordered]@{
            timestamp = $evt.timestamp
            source    = "telemetry"
            type      = $evt.type
            name      = $evt.name
            message   = $evt.message
        })
    }
}

$sortedTimeline = @($timelineEntries | Sort-Object { [datetime]$_.timestamp })

Write-Status "Merged timeline: $($sortedTimeline.Count) entries."

$result = [ordered]@{
    jobKey              = $JobKey
    environment         = $EnvName
    runtimeIdentifier   = $RuntimeIdentifier
    executionInstanceId = $ExecutionInstanceId
    resourceGroup       = $TargetRG
    workspace           = $WorkspaceName
    runs                = $runsOutput
    logCount            = $logCount
    truncated           = $truncated
    timeline            = $sortedTimeline
}

if ($agentTelemetryOutput) {
    $result['agentTelemetry'] = [ordered]@{
        operationIds = $agentTelemetryOutput.operationIds
        eventCount   = $agentTelemetryOutput.eventCount
    }
}

# Single JSON blob to stdout -- the only pipeline output
$result | ConvertTo-Json -Depth 10 -Compress:$false
