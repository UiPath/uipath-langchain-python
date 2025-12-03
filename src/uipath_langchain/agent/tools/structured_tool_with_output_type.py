from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field
from typing_extensions import override


class StructuredToolWithOutputType(StructuredTool):
    output_type: Any = Field(Any, description="Output type.")

    @override
    @property
    def OutputType(self) -> type[Any]:
        return self.output_type
