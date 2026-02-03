from typing import Any

from pydantic import Field
from typing_extensions import override

from .base_uipath_structured_tool import BaseUiPathStructuredTool


class StructuredToolWithOutputType(BaseUiPathStructuredTool):
    output_type: Any = Field(Any, description="Output type.")

    @override
    @property
    def OutputType(self) -> type[Any]:
        return self.output_type
