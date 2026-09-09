"""Base model for every runtime-generated schema class.

`datamodel-code-generator` emits classes deriving from a configurable base. Every
model generated from a tool or agent schema derives from
:class:`UiPathDatamodelCodeGeneratorBaseModel`, which supplies the two
configuration options the runtime depends on:

* ``serialize_by_alias`` -- properties whose JSON names are not valid Python
  identifiers are generated as sanitized fields carrying an alias. Serializing by
  alias is what puts the original JSON names back on the wire, so a tool call
  reaches Integration Service with the property names its schema declared.
* ``extra="allow"`` -- the default for a schema that does not say otherwise. A
  schema with ``additionalProperties: false`` generates its own
  ``model_config``, and Pydantic merges that over this one, so ``extra`` still
  ends up ``"forbid"`` there.

The base also makes a declared JSON property name usable as an attribute, which
is the half of that contract sanitizing the field name would otherwise break.
See :meth:`UiPathDatamodelCodeGeneratorBaseModel.__getattr__`.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class UiPathDatamodelCodeGeneratorBaseModel(BaseModel):
    """Base class for models generated from JSON Schema at runtime."""

    model_config = ConfigDict(serialize_by_alias=True, extra="allow")

    def __getattr__(self, name: str) -> Any:
        """Resolve a declared JSON property name to its sanitized field.

        Serializing by alias makes ``model_dump()`` alias-keyed, and consumers read
        the dumped keys straight back off the instance: LangChain's
        ``BaseTool._parse_input`` builds its kwargs with ``getattr(result, key)``
        for every dumped key. Under the legacy backend the JSON name *was* the
        field name, so that resolved; here the field is sanitized, so a property
        declared ``Content-Type`` would raise ``AttributeError`` mid-tool-call.

        Accepting the alias restores what callers already relied on -- the name the
        schema declared works as an attribute -- so alias-unaware consumers behave
        as they did before.

        Only reached when normal lookup fails, so real fields, methods and extras
        are untouched.
        """
        try:
            return super().__getattr__(name)  # type: ignore[misc]
        except AttributeError:
            pass

        for field_name, field in type(self).model_fields.items():
            if field.alias == name and field_name != name:
                return getattr(self, field_name)

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )
