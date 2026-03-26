from pydantic import SecretStr
from uipath_langchain_client.settings import PlatformSettings

# Use model_construct to bypass the model_validator that triggers authentication
agent_hub_dummy_settings = PlatformSettings.model_construct(
    base_url="https://alpha.uipath.com/TestOrg/TestTenant",
    organization_id="TestOrg",
    tenant_id="TestTenant",
    access_token=SecretStr("TestAccessToken"),
)
