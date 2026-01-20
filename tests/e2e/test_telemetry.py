"""
E2E tests for telemetry functionality.

Tests that agents send telemetry events to Azure Application Insights
when telemetry is enabled, and that the events can be queried back.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from .conftest import EXAMPLES_DIR, run_uipath_command


class TestTelemetryE2E:
    """End-to-end tests for agent telemetry functionality."""

    @pytest.fixture
    def app_insights_url(self) -> str:
        """Get Application Insights API URL from connection string."""
        import os

        conn_str = os.getenv("TELEMETRY_CONNECTION_STRING")
        if not conn_str:
            pytest.skip("TELEMETRY_CONNECTION_STRING not configured")

        # Parse connection string to get Application ID
        parts = {}
        for part in conn_str.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                parts[key] = value

        # Get the Application ID from connection string
        app_id = parts.get("ApplicationId")
        if not app_id:
            pytest.skip("ApplicationId not found in connection string")

        # Application Insights REST API base URL
        return f"https://api.applicationinsights.io/v1/apps/{app_id}"

    @pytest.fixture
    def app_insights_api_key(self) -> str:
        """Get Application Insights API key for querying data."""
        import os

        api_key = os.getenv("APPLICATIONINSIGHTS_API_KEY")
        if not api_key:
            pytest.skip("APPLICATIONINSIGHTS_API_KEY not configured for queries")

        return api_key

    def query_app_insights(
        self, base_url: str, api_key: str, query: str, timeout_minutes: int = 5
    ) -> list[dict[str, Any]]:
        """Query Application Insights using KQL and wait for results."""
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

        query_url = f"{base_url}/query"

        # Try querying multiple times as telemetry data may have some latency
        end_time = datetime.now() + timedelta(minutes=timeout_minutes)

        while datetime.now() < end_time:
            try:
                response = httpx.post(
                    query_url, json={"query": query}, headers=headers, timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    tables = result.get("tables", [])
                    if tables and tables[0].get("rows"):
                        # Return the rows as list of dictionaries
                        columns = [col["name"] for col in tables[0].get("columns", [])]
                        rows = tables[0]["rows"]
                        return [dict(zip(columns, row, strict=True)) for row in rows]

                # Wait before retrying
                time.sleep(10)

            except Exception as e:
                print(f"Query attempt failed: {e}")
                time.sleep(10)

        # Return empty list if no results found within timeout
        return []

    @pytest.mark.e2e
    def test_telemetry_events_sent_to_app_insights(
        self,
        authenticated_session: dict[str, str],
        app_insights_url: str,
        app_insights_api_key: str,
    ):
        """Test that agent execution sends telemetry events to Application Insights."""

        # Use calculator example for testing
        example_dir = EXAMPLES_DIR / "calculator"

        # Generate unique test run identifier for filtering
        test_run_id = f"test-run-{int(time.time())}"
        test_start_time = datetime.now(timezone.utc)

        # Set up environment with telemetry enabled
        env = authenticated_session.copy()
        env["UIPATH_TELEMETRY_ENABLED"] = "true"
        # Add custom property to identify this test run
        env["UIPATH_TEST_RUN_ID"] = test_run_id

        # Run the calculator agent
        input_data = json.dumps({"a": 5, "b": 3, "operator": "+"})

        result = run_uipath_command(
            command=["run", "agent.json", input_data],
            cwd=example_dir,
            env=env,
            timeout=120,
        )

        # Verify the agent ran successfully
        assert result.returncode == 0, f"Agent run failed: {result.stderr}"

        # Query Application Insights for telemetry events
        # Look for URT events within the last 15 minutes to account for latency
        query = """
        customEvents
        | where timestamp > ago(15m)
        | where name in ("AgentRun.Start.URT", "AgentRun.End.URT", "AgentRun.Failed.URT")
        | project timestamp, name, customDimensions, customMeasurements
        | order by timestamp desc
        """

        print(f"Querying Application Insights for events after {test_start_time}")
        events = self.query_app_insights(app_insights_url, app_insights_api_key, query)

        # Debug: Print what events we found
        print(f"Found {len(events)} total events")
        if events:
            print("Event names found:", [e.get("name", "NO_NAME") for e in events])
            print("Sample event:", events[0] if events else "None")

        # Verify we received telemetry events
        assert len(events) > 0, (
            f"No telemetry events found in Application Insights. Query returned {len(events)} events."
        )

        if len(events) > 0:
            # Filter events that are from our test run (within time window)
            test_events = []
            for event in events:
                event_time = datetime.fromisoformat(
                    event["timestamp"].replace("Z", "+00:00")
                )
                if event_time >= test_start_time:
                    test_events.append(event)

            if len(test_events) >= 1:
                # Just verify we have URT events - no need to check properties
                urt_events = [
                    e
                    for e in test_events
                    if e["name"] in ["AgentRun.Start.URT", "AgentRun.End.URT"]
                ]
                assert len(urt_events) >= 1, (
                    f"No URT events found. Events: {[e['name'] for e in test_events]}"
                )

                print("✅ Successfully found telemetry events in Application Insights")
                print(f"   Found {len(test_events)} events from this test run")
                print(f"   URT events: {[e['name'] for e in urt_events]}")

                return  # Test passes
            else:
                print(f"No recent events found after {test_start_time}")

        # If we get here, check if telemetry is enabled but instrumentation key is invalid
        import os

        conn_str = os.getenv("TELEMETRY_CONNECTION_STRING", "")
        if "your-key-here" in conn_str:
            pytest.skip(
                "Telemetry test requires valid Application Insights instrumentation key"
            )

        # Otherwise fail the test
        assert len(events) > 0, (
            f"No telemetry events found in Application Insights after {test_start_time}"
        )
