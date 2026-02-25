const PLAYGROUND_BASE_URL = `https://${__ENV.CLOUD_ENVIRONMENT}.uipath.com/${__ENV.ORGANIZATION_ID}/${__ENV.TENANT_ID}/agenthub_`;
const ORCH_BASE_URL = `https://${__ENV.CLOUD_ENVIRONMENT}.uipath.com/${__ENV.ORGANIZATION_ID}/${__ENV.TENANT_ID}/orchestrator_`;

export const PLAYGROUND_START_URL = `${PLAYGROUND_BASE_URL}/design/debug`;
export const ORCH_START_URL = `${ORCH_BASE_URL}/odata/Jobs/UiPath.Server.Configuration.OData.StartJobs`;
export const ORCH_GET_JOBS_URL = `${ORCH_BASE_URL}/odata/Jobs`;
export const ORCH_KILL_JOBS_URL = `${ORCH_BASE_URL}/odata/Jobs/UiPath.Server.Configuration.OData.StopJobs`;
export const ORCH_SIGNALR_URL = `${ORCH_BASE_URL}/signalr/robotdebug`;
