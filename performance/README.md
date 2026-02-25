## Run locally

Get `PERF_PASSWORD` from `1Password/Agent Perf Test` vault.

```powershell
$env:PERF_USERNAME='uipath.perf.test@uipath-qa.com'; $env:PERF_PASSWORD=''; $env:MODE='startPlayground'; $env:PROCESS_KEY=''; $env:SOLUTION_PROJECT_ID='48d96482-dfe2-48eb-9361-6eb5fe78bb91'; $env:INPUT_ARGUMENTS='{\"longString\":\"test\"}'; $env:FOLDER_ID='297721'; $env:DURATION='1s'; $env:RATE='1'; $env:TIME_UNIT='1s'; $env:PRE_ALLOCATED_VUS='1200'; $env:MAX_VUS='1200'; $env:CLOUD_ENVIRONMENT='staging'; $env:ORGANIZATION_NAME='testcloudperformance'; $env:ORGANIZATION_ID='c0c241d2-01e4-4c55-82fa-2f70083dd89e'; $env:TENANT_NAME='aop_03_alex_groza'; $env:TENANT_ID='11cb2dc5-c10f-4f0c-a5b1-ac238eb27507'; $env:HALT_WAIT_THRESHOLD='5'; $env:SOCKET_TIMEOUT_MINUTES='5'; ./run.ps1
```


