"""Governance integration for ``uipath-langchain``.

Exposes :class:`GovernanceCallbackHandler` — a LangChain callback
handler that calls an :class:`~uipath.core.adapters.EvaluatorProtocol`
on the model and tool lifecycle. Wired into a run by passing an
``evaluator`` to :class:`UiPathLangGraphRuntimeFactory`; the factory
builds the handler and hands it to the runtime through the existing
``callbacks`` channel.

Importing this module has no side effects: no adapter is registered,
no global state is mutated.
"""

from .callbacks import GovernanceCallbackHandler

__all__ = ["GovernanceCallbackHandler"]
