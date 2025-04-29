# Support Ticket Classification System

Use LangGraph with Azure OpenAI to automatically classify support tickets into predefined categories with confidence scores. UiPath Action Center integration for human approval step.

## Debug

1. Clone the repository:

```bash
git clone
cd samples\ticket-classification
```

2. Install dependencies:

```bash
pip install uv
uv venv -p 3.11 .venv
.venv\Scripts\activate
uv sync
```

3. Create a `.env` file in the project root with the following configuration:

```env
UIPATH_URL=https://alpha.uipath.com/ada/byoa
UIPATH_ACCESS_TOKEN=xxx
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_ENDPOINT=xxx
```

```bash
uipath run <entrypoint> <input> [--resume]
```

### Run

To classify a ticket, run the script using UiPath CLI:

```bash
uipath run agent '{"message": "GET Assets API does not enforce proper permissions Assets.View", "ticket_id": "TICKET-2345"}'
```

### Resume

To resume the graph with approval:

```bash
uipath run agent true --resume
```

### Input Format

The input ticket should be in the following format:

```json
{
    "message": "The ticket message or description",
    "ticket_id": "Unique ticket identifier",
    "assignee"[optional]: "username or email of the person assigned to handle escalations"
}
```

### Output Format

The script outputs JSON with the classification results:

```json
{
    "label": "security",
    "confidence": 0.9
}
```

## Deployment Guide

This guide walks you through deploying and running the ticket classification agent on the UiPath Cloud Platform.

### Prerequisites

-   Access to UiPath Cloud Platform
-   Python 3.11 or higher
-   Git

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/UiPath/uipath-langchain-python.git
cd uipath-langchain-python/samples/ticket-classification
```

### 2. Action App Deployment

The Ticket Classification Agent utilizes HITL (Human In The Loop) technology, allowing the system to incorporate feedback directly from supervisory personnel. We'll leverage UiPath [Action Center](https://docs.uipath.com/action-center/automation-suite/2023.4/user-guide/introduction) for this functionality.

Follow these steps to deploy the pre-built application using [UiPath Solutions Management](https://docs.uipath.com/solutions-management/automation-cloud/latest/user-guide/solutions-management-overview):

1. **Upload Solution Package**

    - Navigate to UiPath Solutions Management
    - Drag and drop [generic-escalation-app-solution-1.0.0.zip](escalation_app_solution/generic-escalation-app-solution-1.0.0.zip) to the upload area
    - Click the _Upload_ button

    ![upload-solution-package](../../docs/sample_images/ticket-classification/upload-solution-package.png)

2. **Initiate Deployment**

    - Wait for the package to be processed and ready for deployment

    ![deploy-solution-package](../../docs/sample_images/ticket-classification/deploy-solution-package.png)

3. **Select Destination**

    - Choose a destination folder or install as root folder under tenant

    ![choose-destination-folder](../../docs/sample_images/ticket-classification/solution-destination-folder.png)

4. **Complete Configuration**

    - Follow the solution configuration wizard prompts

    ![solution-configuration-wizard](../../docs/sample_images/ticket-classification/deploy-solution-package-wizard.png)

5. **Activate the Apps**

    - After deployment, activate the apps following the [UiPath documentation](https://docs.uipath.com/apps/automation-cloud/latest/user-guide/apps-in-solutions-management)

    ![activate-deployment](../../docs/sample_images/ticket-classification/activate-deployment.png)
    ![activate-apps](../../docs/sample_images/ticket-classification/activate-apps.png)

6. **Verify and Configure**

    - Navigate to the solution folder to verify the escalation app creation

    ![navigate-to-solution-folder](../../docs/sample_images/ticket-classification/navigate-to-solution-folder.png)

    - Copy the folder path for configuration

    ![copy-folder-path](../../docs/sample_images/ticket-classification/copy-folder-path.png)

    - Update the `_PLACEHOLDER` string in `main.py` (line 148) with the copied folder path

### 3. Agent Setup and Publishing

1. **Set Up Python Environment**

```bash
# Install UV package manager
pip install uv

# Create and activate virtual environment
uv venv -p 3.11 .venv

# Windows
.venv\Scripts\activate

# Unix-like Systems
source .venv/bin/activate

# Install dependencies
uv sync
```

2. **UiPath Authentication**

```bash
uipath auth
```

> **Note:** After successful authentication in the browser, select the tenant for publishing the agent package.

```
👇 Select tenant:
  0: DefaultTenant
  1: Tenant2
  2: Tenant3
...
Select tenant: 2
```

3. **Package and Publish**

```bash
# Create and publish the package
uipath pack
uipath publish
```

Select the feed to publish your package:

```
👇 Select package feed:
  0: Orchestrator Tenant Processes Feed
  1: Orchestrator Folder1 Feed
  2: Orchestrator Folder2 Feed
  3: Orchestrator Personal Workspace Feed
  ...
Select feed number: 3
```

> Note: When publishing to personal workspace feed, the process will be auto-created for you.

### 4. Running and Monitoring the Agent

1. **Start the Agent**

    - Navigate to your agent in the UiPath workspace
    - Click "Run" to start a new job

    ![run the agent](../../docs/sample_images/ticket-classification/run-agent.png)
    ![start job](../../docs/sample_images/ticket-classification/start-job.png)

2. **Monitor Progress**

    - Track the agent's progress using the details side panel

    ![monitor agent](../../docs/sample_images/ticket-classification/monitor-agent.png)

3. **Handle Human-in-the-Loop Tasks**

    - When a ticket is classified, a _Resume Condition_ tab will appear
    - Use this link to navigate to UiPath Action Center for human intervention

    ![resume condition](../../docs/sample_images/ticket-classification/resume-condition.png)

For detailed information about UiPath Action Center and its features, refer to the [official documentation](https://docs.uipath.com/action-center/automation-suite/2024.10/user-guide/introduction).
