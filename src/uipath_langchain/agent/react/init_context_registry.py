"""Registry for resource types that contribute init-time context.

Resource modules self-register by calling ``register_init_context_provider``
at module level. The INIT node calls ``gather_init_context`` to collect
additional context from all registered providers, without needing to know
which resource types participate.
"""

import logging
from typing import Protocol, Sequence

from uipath.agent.models.agent import BaseAgentResourceConfig

logger = logging.getLogger(__name__)


class InitContextProvider(Protocol):
    """Contract for a resource type's init-time context builder."""

    async def __call__(
        self,
        resources: Sequence[BaseAgentResourceConfig],
    ) -> str | None: ...


_registry: dict[str, InitContextProvider] = {}


def register_init_context_provider(
    name: str,
    provider: InitContextProvider,
) -> None:
    """Register a provider that contributes init-time context.

    Args:
        name: Identifier for logging and deduplication.
        provider: Async callable matching ``InitContextProvider``.
    """
    if name in _registry:
        raise ValueError(f"Init context provider '{name}' is already registered")
    _registry[name] = provider
    logger.debug("Registered init context provider: %s", name)


async def gather_init_context(
    resources: Sequence[BaseAgentResourceConfig],
) -> str | None:
    """Call all registered providers and merge their context contributions.

    Args:
        resources: The agent's resource configs.

    Returns:
        Merged context string, or None if no provider contributed.
    """
    parts: list[str] = []
    for name, provider in _registry.items():
        try:
            result = await provider(resources)
            if result:
                parts.append(result)
                logger.info(
                    "Init context provider '%s' contributed %d chars",
                    name,
                    len(result),
                )
        except Exception:
            logger.exception("Init context provider '%s' failed; skipping", name)
    return "\n\n".join(parts) if parts else None
