from enum import StrEnum


class LlmGatewayHeaders(StrEnum):
    """LLM Gateway headers."""

    IS_BYO_EXECUTION = "x-uipath-llmgateway-isbyoexecution"
    EXECUTION_DEPLOYMENT_TYPE = "x-uipath-llmgateway-executiondeploymenttype"
    IS_PII_MASKED = "x-uipath-llmgateway-ispiimasked"
