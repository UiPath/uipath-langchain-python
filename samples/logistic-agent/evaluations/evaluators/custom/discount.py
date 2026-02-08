import json
from typing import Optional

from uipath.eval.evaluators import BaseEvaluator, BaseEvaluationCriteria, BaseEvaluatorConfig
from uipath.eval.models import AgentExecution, EvaluationResult, NumericEvaluationResult, ErrorEvaluationResult
from opentelemetry.sdk.trace import ReadableSpan


class DiscountEvaluationCriteria(BaseEvaluationCriteria):
    """Evaluation criteria for the Discount evaluator."""

    expected_discount: Optional[float] = None
    """Expected discount percentage. If None, evaluator will check against discount rules programmatically."""


class DiscountEvaluatorConfig(BaseEvaluatorConfig[DiscountEvaluationCriteria]):
    """Configuration for the Discount evaluator."""

    name: str = "DiscountEvaluator"
    default_evaluation_criteria: Optional[DiscountEvaluationCriteria] = None


class DiscountEvaluator(BaseEvaluator[DiscountEvaluationCriteria, DiscountEvaluatorConfig, None]):
    """Evaluator that checks if the agent calculated the correct discount based on order history.

    The evaluator validates the discount calculation against the discount policy:
    - 7+ orders in last 7 days: 15% discount
    - 5-6 orders in last 7 days: 10% discount
    - 3-4 orders in last 7 days: 5% discount
    - Less than 3 orders: 0% discount

    If expected_discount is provided, checks strictly against that value.
    If not provided, accepts a +/- 1% difference to account for edge cases.
    """

    def extract_discount_from_spans(self, agent_trace: list[ReadableSpan]) -> tuple[float, int]:
        """Extract the calculated discount and order count from the trace.

        Args:
            agent_trace: List of OpenTelemetry spans from agent execution

        Returns:
            Tuple of (discount_percentage, orders_in_last_7_days)

        Raises:
            Exception: If no 'calculate_discount' span is found
        """
        for span in agent_trace:
            if span.name == "calculate_discount":
                if span.attributes:
                    # Extract the output value
                    output_value_as_str = span.attributes.get("output.value", "{}")
                    assert isinstance(output_value_as_str, str)
                    output_value = json.loads(output_value_as_str)

                    discount_percentage = output_value.get("discount_percentage", 0.0)
                    orders_count = output_value.get("orders_in_last_7_days", 0)

                    return float(discount_percentage), int(orders_count)

        raise Exception("No 'calculate_discount' span found in agent trace")

    def calculate_expected_discount(self, orders_count: int) -> float:
        """Calculate the expected discount based on order count and discount policy.

        Args:
            orders_count: Number of orders in the last 7 days

        Returns:
            Expected discount percentage (0.0, 5.0, 10.0, or 15.0)
        """
        if orders_count >= 7:
            return 15.0
        elif orders_count >= 5:
            return 10.0
        elif orders_count >= 3:
            return 5.0
        else:
            return 0.0

    @classmethod
    def get_evaluator_id(cls) -> str:
        """Get the evaluator ID."""
        return "DiscountEvaluator"

    async def evaluate(
        self,
        agent_execution: AgentExecution,
        evaluation_criteria: DiscountEvaluationCriteria
    ) -> EvaluationResult:
        """Evaluate the agent's discount calculation against the criteria.

        Args:
            agent_execution: The execution details containing:
                - agent_input: The input received by the agent
                - agent_output: The actual output from the agent
                - agent_trace: The execution trace from the agent (list of OpenTelemetry spans)
                - simulation_instructions: The simulation instructions for the agent
            evaluation_criteria: The criteria to evaluate against

        Returns:
            NumericEvaluationResult with score 1.0 if correct, 0.0 if incorrect
            ErrorEvaluationResult if evaluation fails
        """
        try:
            # Extract discount and order count from trace
            actual_discount, orders_count = self.extract_discount_from_spans(
                agent_execution.agent_trace
            )

            # Determine expected discount
            if evaluation_criteria.expected_discount is not None:
                # Strict comparison against provided expected discount
                expected_discount = evaluation_criteria.expected_discount
                is_correct = actual_discount == expected_discount

                return NumericEvaluationResult(
                    score=1.0 if is_correct else 0.0,
                    details=f"Expected discount: {expected_discount}%, Actual discount: {actual_discount}%. {'Match!' if is_correct else 'Mismatch!'}"
                )
            else:
                # Calculate expected discount based on order count
                expected_discount = self.calculate_expected_discount(orders_count)

                # Accept +/- 1% difference
                difference = abs(actual_discount - expected_discount)
                is_correct = difference <= 1.0

                return NumericEvaluationResult(
                    score=1.0 if is_correct else 0.0,
                    details=f"Orders in last 7 days: {orders_count}. Expected discount: {expected_discount}%, Actual discount: {actual_discount}%. Difference: {difference}%. {'Within acceptable range (+/- 1%)' if is_correct else 'Outside acceptable range'}"
                )

        except Exception as e:
            return ErrorEvaluationResult(
                error=f"Failed to evaluate discount: {str(e)}"
            )
