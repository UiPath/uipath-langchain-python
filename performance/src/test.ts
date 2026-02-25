// noinspection JSUnusedGlobalSymbols

import http from 'k6/http';
import {check, sleep} from 'k6';
import {IEnvironment, LoginHandler, SUT} from '@uipath/k6-login';
import {
    ORCH_GET_JOBS_URL,
    ORCH_KILL_JOBS_URL, ORCH_SIGNALR_URL,
    ORCH_START_URL,
    PLAYGROUND_START_URL
} from "./config/urls";
import ws from "k6/ws";
import {Trend} from "k6/metrics";

const onExecutionStartedLatency = new Trend('on_execution_started_latency')

const staging: IEnvironment = {
    sut: __ENV.CLOUD_ENVIRONMENT == 'staging' ? SUT.staging : SUT.alpha,
    username: __ENV.PERF_USERNAME,
    password: __ENV.PERF_PASSWORD,
    organization: __ENV.ORGANIZATION_NAME,
    tenant: __ENV.TENANT_NAME,
};

export const options = {
    scenarios: {
        startAgents: {
            exec: __ENV.EXEC,
            executor: 'ramping-arrival-rate',
            preAllocatedVUs: parseInt(__ENV.PRE_ALLOCATED_VUS),
            maxVUs: parseInt(__ENV.MAX_VUS),
            timeUnit: __ENV.TIME_UNIT,
            stages: [
                {duration: '30s', target: __ENV.RATE},
                {duration: __ENV.DURATION, target: __ENV.RATE},
                {duration: '30s', target: 0}
            ]
        }
    },
    thresholds: {
        http_req_failed: ['rate<0.01'],
        'http_req_failed{name:ProjectConfiguration}': ['rate<0.01'],
        'http_req_failed{name:DebugStart}': ['rate<0.01'],
        'http_req_failed{name:UnattendedStart}': ['rate<0.01'],
        'http_req_failed{name:OrchestratorNegociateSignalR}': ['rate<0.01'],
        'http_req_failed{name:AzureNegociateSignalR}': ['rate<0.01'],
        'http_req_duration{name:ProjectConfiguration}': ['min<1000000'],
        'http_req_duration{name:DebugStart}': ['min<1000000'],
        'http_req_duration{name:UnattendedStart}': ['min<1000000'],
        'http_req_duration{name:OrchestratorNegociateSignalR}': ['min<1000000'],
        'http_req_duration{name:AzureNegociateSignalR}': ['min<1000000']
    },
    setupTimeout: '30m',
    teardownTimeout: '30m',
};

export function setup() {
    const login = new LoginHandler(true);

    const portalData = login.portalUserAndPasswordLogin(staging);
    const uberData = login.uberClientLogin(staging, portalData);

    const playgroundHeaders = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${uberData.accessToken}`,
        'X-Uipath-Accountid': `${__ENV.ORGANIZATION_ID}`,
        'X-Uipath-Tenantid': `${__ENV.TENANT_ID}`
    };

    const orchestratorHeaders = {
        'Content-Type': 'application/json',
        'X-UIPATH-OrganizationUnitId': __ENV.FOLDER_ID,
        'Authorization': `Bearer ${uberData.accessToken}`
    };

    killAllJobs(orchestratorHeaders);

    return {
        playgroundHeaders,
        orchestratorHeaders,
        uberDataAccessToken: uberData.accessToken
    }
}

export function startPlayground(data) {
    if (!data) {
        console.error(`Setup failed. Abort execution.`)
        return;
    }

    const startResponse = http.post(PLAYGROUND_START_URL,
        JSON.stringify({
            Entrypoint: 'agent.json',
            InputArguments: __ENV.INPUT_ARGUMENTS,
            JobType: "PythonAgent",
            SourceProjectId: __ENV.SOLUTION_PROJECT_ID,
            TargetRuntime: "pythonAgent"
        }),
        {
            headers: data.playgroundHeaders,
            tags: {
                name: 'DebugStart'
            }
        }
    );

    check(startResponse, {
        'successful playground start': (r) => r.status === 200
    });

    if (startResponse.status !== 200) {
        console.error(`Request failed with status ${startResponse.status}:`, {
            sourceProjectId: __ENV.SOURCE_PROJECT_ID,
            response: startResponse.body,
            headers: startResponse.headers
        });
        throw new Error('Failed to start agent.');
    }


    const jobKey = JSON.parse(startResponse.body as string).jobKey;
    const jobStartTimestamp = Date.now();
    console.log(`[JobId=${jobKey}] Job started`);

    const connectionDetails = negotiateRobotDebugConnection(
        ORCH_SIGNALR_URL,
        jobKey,
        data.uberDataAccessToken
    );

    connectToRobotDebug(connectionDetails, jobKey, jobStartTimestamp);
}

export function negotiateRobotDebugConnection(
    robotDebugUrl: string,
    sessionId: string,
    accessToken: string
): { url: string; accessToken: string; connectionToken: string } {

    const orchestratorNegotiateUrl = `${robotDebugUrl}/negotiate?sessionId=${sessionId}&negotiateVersion=1`;

    const orchestratorResponse = http.post(orchestratorNegotiateUrl, null, {
        headers: {
            'Authorization': `Bearer ${accessToken}`,
            'Content-Length': '0',
            'x-signalr-user-agent': 'k6-signalr-client/1.0'
        },
        tags: {
            name: 'OrchestratorNegociateSignalR'
        }
    });

    check(orchestratorResponse, {
        'successful orchestrator negociation': (r) => r.status === 200
    });
    if (orchestratorResponse.status !== 200) {
        throw new Error(
            `[SessionId=${sessionId}] Orchestrator negotiation failed: ${orchestratorResponse.status} - ${orchestratorResponse.body}`
        );
    }

    const orchestratorData = JSON.parse(orchestratorResponse.body as string);
    const azureSignalRUrl = orchestratorData.url;
    const azureAccessToken = orchestratorData.accessToken;
    const azureNegotiateUrl = azureSignalRUrl.replace('/client/?', '/client/negotiate?');

    const azureResponse = http.post(azureNegotiateUrl, null, {
        headers: {
            'Authorization': `Bearer ${azureAccessToken}`,
            'Content-Length': '0'
        },
        tags: {
            name: 'AzureNegociateSignalR'
        }
    });

    check(azureResponse, {
        'successful azure signalr negociation': (r) => r.status === 200
    });
    if (azureResponse.status !== 200) {
        throw new Error(
            `[SessionId=${sessionId}] Azure SignalR negotiation failed: ${azureResponse.status} - ${azureResponse.body}`
        );
    }

    const azureData = JSON.parse(azureResponse.body as string);

    return {
        url: azureSignalRUrl,
        accessToken: azureAccessToken,
        connectionToken: azureData.connectionToken
    };
}

export function connectToRobotDebug(
    connectionDetails: { url: string; accessToken: string; connectionToken: string },
    sessionId: string,
    jobStartTimestamp: number
): void {
    const timeout: number = parseInt(__ENV.SOCKET_TIMEOUT_MINUTES) * 60 * 1000;

    const wssUrl = connectionDetails.url.replace('https://', 'wss://') +
        `&id=${connectionDetails.connectionToken}&access_token=${connectionDetails.accessToken}`;

    let isHandshakeComplete = false;
    let invocationId = 0;

    const wsResponse = ws.connect(wssUrl, {}, function (socket) {

        socket.on('open', function () {
            console.log(`[SessionId=${sessionId}] WebSocket opened, sending handshake`);
            socket.send('{"protocol":"json","version":1}\x1e');
        });

        socket.on('message', (message: string) => {
            if (message.endsWith('\x1e')) {
                message = message.slice(0, -1);
            }

            if (!message || message === '{}') {
                if (!isHandshakeComplete) {
                    console.log(`[SessionId=${sessionId}] Handshake complete`);
                    isHandshakeComplete = true;

                    invocationId++;
                    const startMessage = JSON.stringify({
                        type: 1,
                        target: 'SendCommand',
                        arguments: ['Start', '{"breakpoints":[],"enableStepMode":false}'],
                        invocationId: invocationId.toString()
                    }) + '\x1e';

                    socket.send(startMessage);
                    console.log(`[SessionId=${sessionId}] Start command sent (invocationId: ${invocationId})`);
                }
                return;
            }

            try {
                const data = JSON.parse(message);

                if (data.type === 1) {
                    const target = data.target;

                    if (target === 'OnExecutionStarted') {
                        onExecutionStartedLatency.add(Date.now() - jobStartTimestamp);
                        console.log(`[SessionId=${sessionId}] Execution started. Sending Resume command...`);

                        invocationId++;
                        const resumeMessage = JSON.stringify({
                            type: 1,
                            target: 'SendCommand',
                            arguments: ['Resume', '{"breakpoints":[]}'],
                            invocationId: invocationId.toString()
                        }) + '\x1e';

                        socket.send(resumeMessage);
                        console.log(`[SessionId=${sessionId}] Resume command sent (invocationId: ${invocationId})`);
                        socket.close();
                    }
                }

                else if (data.type === 7) {
                    console.log(`[SessionId=${sessionId}] Server closing connection: ${data.error || 'no error'}`);
                    socket.close();
                }

            } catch (error) {
                console.error(`[SessionId=${sessionId}] Error parsing message: ${error}`);
                console.error(`[SessionId=${sessionId}] Raw message: ${message}`);
                socket.close();
            }
        });

        socket.on('error', (e) => {
            console.error(`[SessionId=${sessionId}] WebSocket error: ${e.error()}`);
            socket.close();
        });

        socket.on('close', () => {
            console.log(`[SessionId=${sessionId}] WebSocket connection closed`);
        });

        socket.setTimeout(() => {
            console.error(`[SessionId=${sessionId}] Timeout after ${timeout}ms - Resume command not sent`);
            socket.close();
        }, timeout);
    });

    if (wsResponse.status !== 101) {
        throw new Error(`[SessionId=${sessionId}] WebSocket connection failed: ${wsResponse.status}`);
    }

    console.log(`[SessionId=${sessionId}] WebSocket connection established (status 101)`);
}

export function startUnattended(data) {
    if (!data) {
        console.error(`Setup failed. Abort execution.`)
        return;
    }

    const response = http.post(ORCH_START_URL,
        JSON.stringify({
            startInfo: {
                ReleaseKey: __ENV.PROCESS_KEY,
                JobPriority: null,
                SpecificPriorityValue: 45,
                AutopilotForRobots: null,
                RunAsMe: false,
                EnvironmentVariables: '',
                InputArguments: __ENV.INPUT_ARGUMENTS,
            }
        }),
        {
            headers: data.orchestratorHeaders,
            tags: {
                name: 'UnattendedStart'
            }
        }
    );

    check(response, {
        'successful unattended start': (r) => r.status === 201
    });
}

function killAllJobs(headers) {
    let noJobsToKillCounter = 0;
    const maxNoToJobsKillAttempts = 3;

    while (noJobsToKillCounter < maxNoToJobsKillAttempts) {
        const query = `$filter=((State eq 'Pending') or (State eq 'Running'))&$select=Id,Key,State,CreationTime&$top=1000&$skip=0&$orderby=Id,Key,State,CreationTime`;
        const jobsUrl = `${ORCH_GET_JOBS_URL}?${encodeURIComponent(query)}`;

        const getJobsResponse = http.get(jobsUrl, {
            headers: headers
        });

        if (getJobsResponse.status !== 200) {
            console.error('Failed to get jobs', {
                url: jobsUrl,
                response: getJobsResponse.body,
                headers: getJobsResponse.headers
            });
            continue;
        }

        const jobsResponseBody = JSON.parse(getJobsResponse.body as string);
        const jobIds = jobsResponseBody.value.map(job => job.Id);

        if (jobIds.length === 0) {
            noJobsToKillCounter++;
            sleep(5);
            continue;
        }
        noJobsToKillCounter = 0;

        const killJobsResponse = http.post(ORCH_KILL_JOBS_URL, JSON.stringify({
            strategy: "Kill",
            jobIds: jobIds
        }), {
            headers: headers
        });

        if (killJobsResponse.status !== 200) {
            console.error('Failed to kill jobs', {
                url: jobsUrl,
                response: getJobsResponse.body,
                headers: getJobsResponse.headers
            });
        }
    }
}

export function handleSummary(data) {
    return {
        'summary.json': JSON.stringify(data, null, 2),
    }
}

