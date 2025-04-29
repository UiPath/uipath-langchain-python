# Human In The Loop

Guide for **Human-In-The-Loop** scenarios within the UiPath-Langchain integration.
It focuses on the **interrupt(model)** functionality, illustrating its role as a symbolic representation of an agent's
wait state within the Langchain framework.

## Models Overview

### 1. CreateAction

The `CreateAction` model is utilized to create an escalation action within the UiPath Action Center as part of an interrupt context. The action will rely on a previously created UiPath app.
After addressing the escalation, the current agent will resume execution.
For more information on UiPath apps, refer to the [UiPath Apps User Guide](https://docs.uipath.com/apps/automation-cloud/latest/user-guide/introduction).

#### Attributes:

-   **name** (Optional[str]): The name of the app.
-   **key** (Optional[str]): The key of the app.
-   **title** (str): The title of the action to create.
-   **data** (Optional[Dict[str, Any]]): Values that the action will be populated with.
-   **app_version** (Optional[int]): The version of the app (defaults to 1).
-   **assignee** (Optional[str]): The username or email of the person assigned to handle the escalation.

#### Example:

```python
from uipath.models import CreateAction
action_output = interrupt(CreateAction(name="AppName", title="Escalate Issue", data={"key": "value"}, app_version=1, assignee="user@example.com"))
```

For a practical implementation of the `CreateAction` model, refer to the [ticket-classification sample](https://github.com/UiPath/uipath-langchain-python/tree/main/samples/ticket-classification). This sample demonstrates how to create an action with dynamic input.

---

### 2. WaitAction

The `WaitAction` model is used to wait for an action to be handled. This model is intended for scenarios where the action has already been created.

#### Attributes:

-   **action** (Action): The instance of the action to wait for.

#### Example:

```python
from uipath.models import WaitAction
action_output = interrupt(WaitAction(action=my_action_instance))
```

---

> 💡 UiPath Langchain sdk also supports **Robot/Agent-in-the-loop** scenarios. In this context, the execution of one agent
> can be suspended until another robot or agent finishes its execution.

### 3. InvokeProcess

The `InvokeProcess` model is utilized to invoke a process within the UiPath cloud platform.
This process can be of various types, including API workflows, Agents or RPA automation.
Upon completion of the invoked process, the current agent will automatically resume execution.

#### Attributes:

-   **name** (str): The name of the process to invoke.
-   **input_arguments** (Optional[Dict[str, Any]]): A dictionary containing the input arguments required for the invoked process.

#### Example:

```python
from uipath.models import InvokeProcess
process_output = interrupt(InvokeProcess(name="MyProcess", input_arguments={"arg1": "value1"}))
```

For a practical implementation of the `InvokeProcess` model, refer to the [multi-agent-planner-researcher-coder-distributed sample](https://github.com/UiPath/uipath-langchain-python/tree/main/samples/multi-agent-planner-researcher-coder-distributed). This sample demonstrates how to invoke a process with dynamic input arguments, showcasing the integration of the interrupt functionality within a multi-agent system or a system where an agent integrates with RPA processes and API workflows.

---

### 4. WaitJob

The `WaitJob` model is used to wait for a job completion. Unlike `InvokeProcess`, which automatically creates a job, this model is intended for scenarios where
the job has already been created.

#### Attributes:

-   **job** (Job): The instance of the job that the agent will wait for. This should be a valid job object that has been previously created.

#### Example:

```python
from uipath.models import WaitJob
job_output = interrupt(WaitJob(job=my_job_instance))
```
