"""BTS attribute builders for transactions and operations.

Each builder returns a flat Dict[str, str] sent to the Automation Tracker API.
"""

from typing import Optional


def build_transaction_attributes() -> dict[str, str]:
    """Build attributes for a BTS transaction."""
    return {
        "Type": "AgentRun",
        "XUiPathImplicitRecord": "True",
    }


def build_common_operation_attributes(
    *,
    job_key: Optional[str] = None,
    tool_type: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes common to all BTS operations."""
    attrs: dict[str, str] = {
        "XUiPathSourceType": "Agent",
        "XUiPathImplicitRecord": "True",
    }
    if job_key:
        attrs["XUiPathSourceKey"] = job_key
    if tool_type:
        attrs["XUiPathToolType"] = tool_type
    return attrs


def build_agent_operation_attributes(
    *,
    job_key: Optional[str] = None,
    process_name: Optional[str] = None,
    process_key: Optional[str] = None,
    package_version: Optional[str] = None,
    package_id: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes for an agent-run BTS operation."""
    attrs = build_common_operation_attributes(job_key=job_key)
    if process_name:
        attrs["XUiPathProcessName"] = process_name
    if process_key:
        attrs["XUiPathProcessKey"] = process_key
    if package_version is not None:
        attrs["XUiPathPackageVersion"] = package_version
    if package_id is not None:
        attrs["XUiPathPackageId"] = package_id
    return attrs


def build_process_tool_attributes(
    *,
    job_key: Optional[str] = None,
    tool_type: Optional[str] = None,
    process_name: Optional[str] = None,
    wait_for_job_key: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes for a process-tool BTS operation."""
    attrs = build_common_operation_attributes(job_key=job_key, tool_type=tool_type)
    if process_name:
        attrs["XUiPathProcessName"] = process_name
    if wait_for_job_key:
        attrs["XUiPathWaitForJobKey"] = wait_for_job_key
    return attrs


def build_agent_tool_attributes(
    *,
    job_key: Optional[str] = None,
    tool_type: Optional[str] = None,
    wait_for_agent_job_key: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes for an agent-tool BTS operation."""
    attrs = build_common_operation_attributes(job_key=job_key, tool_type=tool_type)
    if wait_for_agent_job_key:
        attrs["XUiPathWaitForAgentJobKey"] = wait_for_agent_job_key
    return attrs


def build_context_grounding_tool_attributes(
    *,
    job_key: Optional[str] = None,
    tool_type: Optional[str] = None,
    index_id: Optional[str] = None,
    index_name: Optional[str] = None,
    context_retrieval_mode: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes for a context-grounding-tool BTS operation."""
    attrs = build_common_operation_attributes(job_key=job_key, tool_type=tool_type)
    if index_id:
        attrs["XUiPathIndexId"] = index_id
    if index_name:
        attrs["XUiPathIndexName"] = index_name
    if context_retrieval_mode:
        attrs["XUiPathContextRetrievalMode"] = context_retrieval_mode
    return attrs


def build_escalation_tool_attributes(
    *,
    job_key: Optional[str] = None,
    tool_type: Optional[str] = None,
    task_key: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes for an escalation-tool BTS operation."""
    attrs = build_common_operation_attributes(job_key=job_key, tool_type=tool_type)
    if task_key:
        attrs["XUiPathTaskKey"] = task_key
    return attrs


def build_integration_tool_attributes(
    *,
    job_key: Optional[str] = None,
    tool_type: Optional[str] = None,
    connector_key: Optional[str] = None,
    connector_name: Optional[str] = None,
) -> dict[str, str]:
    """Build attributes for an integration-service-tool BTS operation."""
    attrs = build_common_operation_attributes(job_key=job_key, tool_type=tool_type)
    if connector_key:
        attrs["XUiPathConnectorKey"] = connector_key
    if connector_name:
        attrs["XUiPathConnectorName"] = connector_name
    return attrs
