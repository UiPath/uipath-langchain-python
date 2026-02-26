from uipath_agents._bts.bts_attributes import (
    build_agent_operation_attributes,
    build_common_operation_attributes,
    build_process_tool_attributes,
    build_transaction_attributes,
)


def test_transaction_attributes() -> None:
    attrs = build_transaction_attributes()
    assert attrs["XUiPathImplicitRecord"] == "True"
    assert attrs["Type"] == "AgentRun"


def test_common_operation_attributes() -> None:
    attrs = build_common_operation_attributes(job_key="abc-123")
    assert attrs["XUiPathSourceKey"] == "abc-123"
    assert attrs["XUiPathSourceType"] == "Agent"
    assert attrs["XUiPathImplicitRecord"] == "True"


def test_common_operation_attributes_no_job_key() -> None:
    attrs = build_common_operation_attributes(job_key=None)
    assert "XUiPathSourceKey" not in attrs
    assert attrs["XUiPathSourceType"] == "Agent"


def test_common_operation_attributes_with_tool_type() -> None:
    attrs = build_common_operation_attributes(job_key="j1", tool_type="process")
    assert attrs["XUiPathToolType"] == "process"
    assert attrs["XUiPathSourceKey"] == "j1"


def test_common_operation_attributes_no_tool_type() -> None:
    attrs = build_common_operation_attributes()
    assert "XUiPathToolType" not in attrs


def test_agent_operation_attributes() -> None:
    attrs = build_agent_operation_attributes(
        job_key="j1",
        process_name="MyAgent",
        process_key="pk-1",
        package_version="1.0.0",
        package_id="pid-1",
    )
    assert attrs["XUiPathProcessName"] == "MyAgent"
    assert attrs["XUiPathProcessKey"] == "pk-1"
    assert attrs["XUiPathPackageVersion"] == "1.0.0"
    assert attrs["XUiPathPackageId"] == "pid-1"
    assert attrs["XUiPathSourceKey"] == "j1"


def test_agent_operation_attributes_empty_strings() -> None:
    """Empty string package_version/package_id should still be included."""
    attrs = build_agent_operation_attributes(
        package_version="",
        package_id="",
    )
    assert attrs["XUiPathPackageVersion"] == ""
    assert attrs["XUiPathPackageId"] == ""


def test_agent_operation_attributes_none_omitted() -> None:
    """None package_version/package_id should be omitted."""
    attrs = build_agent_operation_attributes()
    assert "XUiPathPackageVersion" not in attrs
    assert "XUiPathPackageId" not in attrs


def test_process_tool_attributes() -> None:
    attrs = build_process_tool_attributes(
        job_key="j1",
        process_name="InvoiceProcess",
        wait_for_job_key="wjk-1",
    )
    assert attrs["XUiPathProcessName"] == "InvoiceProcess"
    assert attrs["XUiPathWaitForJobKey"] == "wjk-1"


def test_process_tool_attributes_with_tool_type() -> None:
    attrs = build_process_tool_attributes(
        job_key="j1",
        tool_type="process",
        process_name="P",
    )
    assert attrs["XUiPathToolType"] == "process"
    assert attrs["XUiPathSourceKey"] == "j1"
