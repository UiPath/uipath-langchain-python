from pydantic import SecretStr
from uipath_langchain_client.settings import PlatformSettings

# PlatformSettings.validate_environment only decodes the middle segment of the
# access token and checks for an `exp` claim. `e30` is base64url for `{}` — an
# empty payload with no `exp` — so the validator accepts this placeholder
# without any real credential material.
_PLACEHOLDER_ACCESS_TOKEN = "PLACEHOLDER.e30.PLACEHOLDER"

agent_hub_dummy_settings = PlatformSettings(
    base_url="https://alpha.uipath.com/TestOrg/TestTenant",
    organization_id="TestOrg",
    tenant_id="TestTenant",
    access_token=SecretStr(_PLACEHOLDER_ACCESS_TOKEN),
)
