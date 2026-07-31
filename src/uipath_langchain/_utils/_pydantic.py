"""Shared Pydantic model utilities."""

from pydantic import AliasChoices, AliasPath, BaseModel


def get_unique_model_field_name(
    preferred_name: str,
    *models: type[BaseModel] | BaseModel | None,
) -> str:
    """Return a deterministic field name unused by the supplied Pydantic models."""
    occupied_names: set[str] = set()

    for model in models:
        if model is None:
            continue
        model_type = model if isinstance(model, type) else type(model)
        occupied_names.update(model_type.model_fields)

        for field in model_type.model_fields.values():
            aliases: list[str | AliasPath] = []
            if field.alias is not None:
                aliases.append(field.alias)
            if isinstance(field.validation_alias, AliasChoices):
                aliases.extend(field.validation_alias.choices)
            elif field.validation_alias is not None:
                aliases.append(field.validation_alias)

            for alias in aliases:
                if isinstance(alias, str):
                    occupied_names.add(alias)
                elif alias.path and isinstance(alias.path[0], str):
                    occupied_names.add(alias.path[0])

    if preferred_name not in occupied_names:
        return preferred_name

    for suffix in range(1, len(occupied_names) + 1):
        candidate = f"{preferred_name}_{suffix}"
        if candidate not in occupied_names:
            return candidate

    raise AssertionError("A unique model field name must exist")
