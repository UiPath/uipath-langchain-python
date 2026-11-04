import os
import tempfile

import pytest
from uipath.runtime import UiPathRuntimeContext

from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory


@pytest.fixture
def factory() -> UiPathLangGraphRuntimeFactory:
    tmpdir = tempfile.mkdtemp()
    return UiPathLangGraphRuntimeFactory(
        UiPathRuntimeContext(
            runtime_dir=tmpdir,
            state_file=os.path.join(tmpdir, "state.db"),
        )
    )


async def test_factory_asks_the_host_for_a_managed_workspace(
    factory: UiPathLangGraphRuntimeFactory,
) -> None:
    settings = await factory.get_settings()

    assert settings is not None
    assert settings.managed_workspace is True
