"""Custom middleware for logging agent input and output."""

import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentState, before_agent, after_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.runtime import Runtime
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from uipath_langchain.guardrails import GuardrailAction

# Set up logging
logger = logging.getLogger(__name__)


@before_agent
async def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    """Log the input before agent execution starts."""
    messages = state.get("messages", [])
    if messages:
        # Get the last human message (the input)
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                logger.info(f"Agent Input: {msg.content}")
                print(f"[LOG] Agent Input: {msg.content}")
                break
    else:
        logger.info("Agent Input: (no messages)")
        print("[LOG] Agent Input: (no messages)")


@after_agent
async def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    """Log the output after agent execution completes."""
    messages = state.get("messages", [])
    if messages:
        # Get the last AI message (the output)
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                logger.info(f"Agent Output: {msg.content}")
                print(f"[LOG] Agent Output: {msg.content}")
                break
    else:
        logger.info("Agent Output: (no messages)")
        print("[LOG] Agent Output: (no messages)")


# Create middleware instances from the decorated functions
LoggingMiddleware = [log_before_agent, log_after_agent]


@dataclass
class CustomFilterAction(GuardrailAction):
    """Custom action that filters/replaces offensive words in tool input.

    This demonstrates how developers can implement their own custom guardrail actions.
    When a violation is detected, this action:
    1. Filters the offensive word from the input data
    2. Logs the violation with details showing original and filtered text
    3. Returns the filtered input data to be used instead

    Args:
        word_to_filter: The word to filter (case-insensitive)
        replacement: Text to replace the filtered word with (default: "***")
    """

    word_to_filter: str
    replacement: str = "***"

    def _filter_text(self, text: str) -> str:
        """Filter the specified word from text.

        Args:
            text: The text to filter

        Returns:
            Filtered text with the word replaced
        """
        if not text:
            return text

        # Use case-insensitive regex to replace the word
        pattern = re.compile(re.escape(self.word_to_filter), re.IGNORECASE)
        return pattern.sub(self.replacement, text)

    def handle_validation_result(
        self,
        result: GuardrailValidationResult,
        input_data: str | dict[str, Any],
        guardrail_name: str,
    ) -> str | dict[str, Any] | None:
        """Handle validation result by filtering the word and returning modified data.

        Args:
            result: The validation result from the guardrails service
            input_data: The data that was validated (string or dictionary).
                This can be tool input (arguments), tool output (result),
                or message content depending on the guardrail scope.
            guardrail_name: The name of the guardrail that triggered

        Returns:
            Filtered data with the offensive word replaced, or None if no filtering needed.
        """
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            # Extract text for logging and filtering
            if isinstance(input_data, str):
                original_text = input_data
                # Filter the text
                filtered_text = self._filter_text(original_text)

                # Log the filtering action
                if self.word_to_filter.lower() in original_text.lower():
                    log_message = (
                        f"Filtered word '{self.word_to_filter}' detected and replaced. "
                        f"Original: {original_text[:100]}{'...' if len(original_text) > 100 else ''} | "
                        f"Filtered: {filtered_text[:100]}{'...' if len(filtered_text) > 100 else ''}"
                    )
                    logger.info(log_message)
                    print(f"[FILTER][{guardrail_name}] {log_message}")

                # Return filtered text
                return filtered_text

            elif isinstance(input_data, dict):
                # Create a copy to avoid modifying the original
                filtered_data = input_data.copy()
                original_text = None
                filtered_text = None

                # Filter text fields (common patterns)
                for key in ["joke", "text", "content", "message", "input", "output"]:
                    if key in filtered_data and isinstance(filtered_data[key], str):
                        if original_text is None:
                            original_text = filtered_data[key]
                        filtered_data[key] = self._filter_text(filtered_data[key])
                        if filtered_text is None:
                            filtered_text = filtered_data[key]

                # Log the filtering action if word was found
                if original_text and self.word_to_filter.lower() in original_text.lower():
                    log_message = (
                        f"Filtered word '{self.word_to_filter}' detected and replaced. "
                        f"Original: {original_text[:100]}{'...' if len(original_text) > 100 else ''} | "
                        f"Filtered: {filtered_text[:100]}{'...' if len(filtered_text) > 100 else ''}"
                    )
                    logger.info(log_message)
                    print(f"[FILTER][{guardrail_name}] {log_message}")

                # Return filtered dict
                return filtered_data
            else:
                # For other types, convert to string, filter, and return as string
                original_text = str(input_data)
                filtered_text = self._filter_text(original_text)

                if self.word_to_filter.lower() in original_text.lower():
                    log_message = (
                        f"Filtered word '{self.word_to_filter}' detected and replaced. "
                        f"Original: {original_text[:100]}{'...' if len(original_text) > 100 else ''} | "
                        f"Filtered: {filtered_text[:100]}{'...' if len(filtered_text) > 100 else ''}"
                    )
                    logger.info(log_message)
                    print(f"[FILTER][{guardrail_name}] {log_message}")

                return filtered_text

        return None
