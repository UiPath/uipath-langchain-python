param(
    [switch][alias("Staging")]$Stg,
    [switch][alias("Alpha")]$Alp,
    [switch][alias("Prod")]$Prd,
    [string]$Ago = "",
    [string]$StartTime = "",
    [string]$EndTime = "",
    [int]$MaxResults = 500,
    [switch]$Help
)

if ($Help) {
    Write-Host @"
Usage: get-alert-failures.ps1 [-Stg|-Alp|-Prd] [-Ago <offset>] [-StartTime <iso> -EndTime <iso>] [-MaxResults <n>]

Queries agents AppInsights for AgentRun.Failed events and returns grouped failure
categories as JSON on stdout. Progress messages go to stderr.

  -Stg                Staging environment
  -Alp                Alpha environment
  -Prd                Production environment (default)
  -Ago                Time offset, e.g. 24h, 2d, 1d12h (default: 24h)
  -StartTime          Explicit UTC start time (ISO 8601), e.g. 2026-02-23T14:43:16Z
  -EndTime            Explicit UTC end time (ISO 8601), e.g. 2026-02-23T14:58:16Z
  -MaxResults         Maximum individual failure rows to return (default: 500)

Time window: Use either -Ago OR -StartTime/-EndTime. If neither, defaults to -Ago 24h.

Output JSON schema:
  {
    "environment": "stg",
    "appInsightsResource": "agents-stg-appins-ne-appins",
    "timeWindow": { "start": "...", "end": "...", "ago": "24h" },
    "totalFailures": 15,
    "uniqueCategories": 4,
    "categories": [
      {
        "errorType": "StopIteration",
        "errorMessage": "ContextGroundingIndex not found",
        "count": 5,
        "jobKeys": ["uuid1", "uuid2"],
        "representativeJobKey": "uuid1",
        "affectedOrgs": ["org1"],
        "affectedTenants": ["tenant1"],
        "regions": ["ne"],
        "firstSeen": "2026-02-23T14:43:20Z",
        "lastSeen": "2026-02-23T14:57:50Z"
      }
    ],
    "failures": [
      {
        "timestamp": "...",
        "jobKey": "...",
        "errorType": "...",
        "errorMessage": "...",
        "errorCategory": "...",
        "organizationId": "...",
        "tenantId": "...",
        "agentRunSource": "...",
        "region": "...",
        "operationId": "..."
      }
    ]
  }

Examples:
  pwsh ./get-alert-failures.ps1 -Stg -Ago 1h
  pwsh ./get-alert-failures.ps1 -Stg -StartTime 2026-02-23T14:43:16Z -EndTime 2026-02-23T14:58:16Z
  pwsh ./get-alert-failures.ps1 -Prd -Ago 7d -MaxResults 1000
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

# --- Time window resolution ---

$useExplicitWindow = $false
$timeWindowOutput = @{}

if ($StartTime -and $EndTime) {
    $useExplicitWindow = $true
    $timeWindowOutput = [ordered]@{ start = $StartTime; end = $EndTime }
    Write-Status "Time window: $StartTime to $EndTime"
}
else {
    if (-not $Ago) { $Ago = "24h" }
    $timeWindowOutput = [ordered]@{ ago = $Ago }
    Write-Status "Time window: ago $Ago"
}

# --- Check required extensions ---

$extensions = az extension list --query "[].name" -o json 2>$null | ConvertFrom-Json
foreach ($ext in @("resource-graph", "application-insights")) {
    if ($extensions -notcontains $ext) {
        Write-JsonError "The '$ext' az extension is required but not installed. Install it with: az extension add --name $ext"
    }
}

# --- Locate Agents AppInsights via Resource Graph ---

$agentsAppInsName = "agents-$EnvName-appins-ne-appins"
Write-Status "Searching for Agents AppInsights resource '$agentsAppInsName'..."

$resourceJson = az graph query -q @"
resources
| where type == 'microsoft.insights/components'
| where name == '$agentsAppInsName'
| project id, subscriptionId, resourceGroup
| limit 1
"@ --query "data[0]" -o json
Assert-LastExitCode "Resource Graph query for Agents AppInsights"

$resourceData = $resourceJson | ConvertFrom-Json
if (-not $resourceData -or -not $resourceData.id) {
    Write-JsonError "Agents AppInsights resource '$agentsAppInsName' not found. Check the environment name."
}

$agentsAppInsId = $resourceData.id

# --- Query AgentRun.Failed events joined with exceptions ---

Write-Status "Querying AgentRun.Failed events..."

# One row per failure: AgentRun.Failed joined with deduplicated exception info
$failuresQuery = @"
let ExecutionPrefix = "An unexpected error occurred during agent execution, please try again later or contact your Administrator.\nError Details:\n";
let StartupPrefix = "An unexpected error occurred during agent startup, please try again later or contact your Administrator.\nError Details:\n";
customEvents
| where name == "AgentRun.Failed" and customDimensions["Runtime"] == "URT"
| extend
    ErrorCategory = tostring(customDimensions["ErrorCategory"]),
    OrganizationId = tostring(customDimensions["CloudOrganizationId"]),
    TenantId = tostring(customDimensions["CloudTenantId"]),
    AgentRunSource = tostring(customDimensions["AgentRunSource"]),
    ErrorMessage = trim(StartupPrefix, trim(ExecutionPrefix, tostring(customDimensions["ErrorMessage"]))),
    JobKey = tostring(customDimensions["JobKey"]),
    Region = cloud_RoleName
| where ErrorCategory !in ("User")
| join kind=leftouter (
    exceptions
    | summarize ErrorType = take_any(innermostType), InnerMessage = take_any(innermostMessage) by operation_Id
) on operation_Id
| project timestamp, JobKey, ErrorType, ErrorMessage, ErrorCategory, OrganizationId, TenantId, AgentRunSource, Region, operation_Id
| order by timestamp desc
| limit $MaxResults
"@

# Build az CLI arguments for the query
$queryArgs = @(
    "monitor", "app-insights", "query",
    "--ids", $agentsAppInsId,
    "--analytics-query", $failuresQuery,
    "-o", "json"
)
if ($useExplicitWindow) {
    $queryArgs += @("--start-time", $StartTime, "--end-time", $EndTime)
}
else {
    $queryArgs += @("--offset", $Ago)
}

$failuresJson = & az @queryArgs
Assert-LastExitCode "Agents AppInsights query for AgentRun.Failed events"

# --- Parse query results ---

$failuresResult = $failuresJson | ConvertFrom-Json
$failureRows = @()
if ($failuresResult.tables -and $failuresResult.tables[0].rows -and $failuresResult.tables[0].rows.Count -gt 0) {
    $cols = $failuresResult.tables[0].columns | ForEach-Object { $_.name }
    $failureRows = @($failuresResult.tables[0].rows | ForEach-Object {
        $row = $_
        $obj = [ordered]@{}
        for ($c = 0; $c -lt $cols.Count; $c++) {
            $obj[$cols[$c]] = $row[$c]
        }
        $obj
    })
}

$totalFailures = $failureRows.Count
Write-Status "Found $totalFailures failure(s)."

# --- Normalize error signatures for grouping ---
# Strips instance-specific data (UUIDs, numeric IDs, URLs) to cluster equivalent failures.

function Get-NormalizedSignature {
    param([string]$ErrorType, [string]$ErrorMessage)

    $et = if ($ErrorType) { $ErrorType } else { "(none)" }
    # Short type name: strip module path
    $shortType = ($et -split '\.')[-1]

    if ($et -like "*EnrichedException*" -or $et -like "*AgentRuntimeError*") {
        # Extract HTTP method, status code, and normalized service path
        $method = if ($ErrorMessage -match 'HTTP Method:\s*(\w+)') { $Matches[1] } else { "?" }
        $status = if ($ErrorMessage -match 'Status Code:\s*(\d+)') { $Matches[1] } else { "?" }
        $path = "?"
        if ($ErrorMessage -match 'Request URL:\s*https?://[^/]+/[^/]+/[^/]+/([^\n?]+)') {
            $path = $Matches[1]
            # Normalize UUIDs and numeric path segments
            $path = $path -replace '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<id>'
            $path = $path -replace '/\d+/', '/<n>/'
            $path = $path -replace '/\d+$', '/<n>'
            # Strip trailing slash
            $path = $path.TrimEnd('/')
        }
        return "$shortType | $method $status $path"
    }

    if ($et -like "*ValidationError*") {
        # Extract the model name and field from "N validation error(s) for ModelName\nfieldName"
        $model = if ($ErrorMessage -match 'validation error[s]? for (\S+)') { $Matches[1] } else { "?" }
        $field = if ($ErrorMessage -match 'validation error[s]? for \S+\s*\n(\S+)') { $Matches[1] } else { "?" }
        return "$shortType | $model.$field"
    }

    if ($et -like "*GraphInterrupt*") {
        return "GraphInterrupt"
    }

    if ($et -like "*ReadTimeout*" -or $et -like "*httpx.ReadTimeout*") {
        return "$shortType | timeout"
    }

    if ($et -like "*ContextOverflowError*") {
        return "$shortType | context overflow"
    }

    # Strip generic error prefixes that hide the actual error details
    $msg = $ErrorMessage
    $msg = $msg -replace '(?s)^An unexpected error occurred during agent (execution|startup), please try again later or contact your Administrator\.\s*Error Details:\s*', ''

    # Generic: use error type + first line of message, normalized
    $firstLine = ($msg -split "`n")[0].Trim()
    $firstLine = $firstLine -replace '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '<id>'
    if ($firstLine.Length -gt 120) { $firstLine = $firstLine.Substring(0, 120) + "..." }
    return "$shortType | $firstLine"
}

# --- Build structured failure list ---

$failures = @($failureRows | ForEach-Object {
    $errorType = $_['ErrorType']
    $errorMessage = $_['ErrorMessage']
    [ordered]@{
        timestamp      = $_['timestamp']
        jobKey         = $_['JobKey']
        errorType      = $errorType
        errorMessage   = $errorMessage
        signature      = (Get-NormalizedSignature -ErrorType $errorType -ErrorMessage $errorMessage)
        errorCategory  = $_['ErrorCategory']
        organizationId = $_['OrganizationId']
        tenantId       = $_['TenantId']
        agentRunSource = $_['AgentRunSource']
        region         = $_['Region']
        operationId    = $_['operation_Id']
    }
})

# --- Group by normalized signature ---

$groups = @{}
foreach ($f in $failures) {
    $key = $f.signature
    if (-not $groups.ContainsKey($key)) {
        $groups[$key] = [System.Collections.Generic.List[object]]::new()
    }
    $groups[$key].Add($f)
}

$categories = @($groups.GetEnumerator() | ForEach-Object {
    $items = $_.Value
    $first = $items[0]
    $jobKeys = @($items | ForEach-Object { $_.jobKey } | Select-Object -Unique)
    $orgs = @($items | ForEach-Object { $_.organizationId } | Where-Object { $_ } | Select-Object -Unique)
    $tenants = @($items | ForEach-Object { $_.tenantId } | Where-Object { $_ } | Select-Object -Unique)
    $regions = @($items | ForEach-Object { $_.region } | Where-Object { $_ } | Select-Object -Unique)
    $timestamps = @($items | ForEach-Object { [datetime]$_.timestamp })
    # Pick the most recent job as representative (freshest container logs)
    $mostRecent = ($items | Sort-Object { [datetime]$_.timestamp } | Select-Object -Last 1)

    [ordered]@{
        signature            = $_.Key
        errorType            = $first.errorType
        sampleErrorMessage   = $first.errorMessage
        count                = $items.Count
        jobKeys              = $jobKeys
        representativeJobKey = $mostRecent.jobKey
        affectedOrgs         = $orgs
        affectedTenants      = $tenants
        regions              = $regions
        firstSeen            = ($timestamps | Sort-Object)[0].ToString("o")
        lastSeen             = ($timestamps | Sort-Object)[-1].ToString("o")
    }
} | Sort-Object { -[int]$_.count })

Write-Status "Grouped into $($categories.Count) unique failure category/categories."

# --- Build output ---

$result = [ordered]@{
    environment        = $EnvName
    appInsightsResource = $agentsAppInsName
    timeWindow         = $timeWindowOutput
    totalFailures      = $totalFailures
    uniqueCategories   = $categories.Count
    categories         = $categories
    failures           = $failures
}

$result | ConvertTo-Json -Depth 10 -Compress:$false
