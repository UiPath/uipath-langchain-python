from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)


class TestAgentRuntimeError:
    def test_basic_construction(self):
        err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Something broke",
            detail="Details here",
        )
        assert err.error_info.title == "Something broke"
        assert err.error_info.code == "AGENT_RUNTIME.UNEXPECTED_ERROR"
        assert err.error_info.category == UiPathErrorCategory.UNKNOWN

    def test_user_category_no_wrap(self):
        err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.TERMINATION_MAX_ITERATIONS,
            title="Max iterations",
            detail="Reached limit",
            category=UiPathErrorCategory.USER,
        )
        assert err.error_info.detail == "Reached limit"
        assert err.error_info.category == UiPathErrorCategory.USER

    def test_system_category_auto_wraps(self):
        err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Error",
            detail="raw error text",
            category=UiPathErrorCategory.SYSTEM,
        )
        assert "raw error text" in err.error_info.detail
        assert (
            "unexpected error occurred during agent execution" in err.error_info.detail
        )

    def test_unknown_category_auto_wraps(self):
        err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Error",
            detail="raw error text",
            category=UiPathErrorCategory.UNKNOWN,
        )
        assert "raw error text" in err.error_info.detail
        assert (
            "unexpected error occurred during agent execution" in err.error_info.detail
        )

    def test_should_wrap_false_overrides_system(self):
        err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Error",
            detail="raw only",
            category=UiPathErrorCategory.SYSTEM,
            should_wrap=False,
        )
        assert err.error_info.detail == "raw only"

    def test_should_wrap_true_overrides_user(self):
        err = AgentRuntimeError(
            code=AgentRuntimeErrorCode.UNEXPECTED_ERROR,
            title="Error",
            detail="raw text",
            category=UiPathErrorCategory.USER,
            should_wrap=True,
        )
        assert (
            "unexpected error occurred during agent execution" in err.error_info.detail
        )


class TestAgentStartupError:
    def test_basic_construction(self):
        err = AgentStartupError(
            code=AgentStartupErrorCode.UNEXPECTED_ERROR,
            title="Startup failed",
            detail="Missing config",
        )
        assert err.error_info.title == "Startup failed"
        assert err.error_info.code == "AGENT_STARTUP.UNEXPECTED_ERROR"
        assert err.error_info.category == UiPathErrorCategory.UNKNOWN

    def test_deployment_category_no_wrap(self):
        err = AgentStartupError(
            code=AgentStartupErrorCode.LLM_INVALID_MODEL,
            title="Invalid model",
            detail="Model not supported",
            category=UiPathErrorCategory.DEPLOYMENT,
        )
        assert err.error_info.detail == "Model not supported"

    def test_system_category_auto_wraps(self):
        err = AgentStartupError(
            code=AgentStartupErrorCode.UNEXPECTED_ERROR,
            title="Error",
            detail="raw startup error",
            category=UiPathErrorCategory.SYSTEM,
        )
        assert "raw startup error" in err.error_info.detail
        assert "unexpected error occurred during agent startup" in err.error_info.detail

    def test_should_wrap_false_overrides_system(self):
        err = AgentStartupError(
            code=AgentStartupErrorCode.UNEXPECTED_ERROR,
            title="Error",
            detail="raw only",
            category=UiPathErrorCategory.SYSTEM,
            should_wrap=False,
        )
        assert err.error_info.detail == "raw only"
