"""``execute_sql`` for the Ontology-service path: statements run on the service.

The model's statement is sent to ``POST /ontology/{id}/sql`` as the request body
(never interpolated into a URL) and the rows come back as records, in the same
shape the Data Fabric executor returns so the inner sub-graph and the outer agent
read both alike. The model writes ``LIMIT``/``OFFSET`` into the statement itself
(the prompt says so), so the client's ``limit``/``offset`` arguments are never
passed: the client appends those clauses and refuses a statement that already
carries one, and both at once would be refused. A rejected statement (the
runtime refuses non-SELECT, unknown objects, too many rows) is returned as an
``error`` so the sub-graph feeds it back to the model for a bounded retry.

Logging stays at ids, timings and counts. Statements are logged at DEBUG only;
rows are never logged.
"""

import logging
import time
from typing import Any

from langchain_core.tools import BaseTool

from ...datafabric_query_tool import DataFabricQueryTool
from ..models import DataFabricExecuteSqlInput
from .ontology_service_client import OntologyServiceClient

# The service's default page (no LIMIT) and its LIMIT cap. A page of exactly one of
# these sizes may have been cut short; until the service reports it, say so.
SERVICE_PAGE_SIZES = (50, 200)

logger = logging.getLogger(__name__)


class OntologyServiceQueryExecutor:
    """Runs one SQL statement against one deployed ontology."""

    def __init__(
        self, client: OntologyServiceClient, ontology_id: str, label: str
    ) -> None:
        self._client = client
        self._ontology_id = ontology_id
        self._label = label

    async def __call__(self, sql_query: str) -> dict[str, Any]:
        logger.debug("Ontology %s execute_sql: %s", self._ontology_id, sql_query)
        started = time.perf_counter()
        try:
            result = await self._client.execute_sql_async(self._ontology_id, sql_query)
        except Exception as e:  # the runtime's refusal is the model's retry hint
            logger.warning(
                "Ontology %s SQL rejected after %d ms: %s",
                self._ontology_id,
                int((time.perf_counter() - started) * 1000),
                e,
            )
            return {
                "records": [],
                "total_count": 0,
                "error": str(e),
                "sql_query": sql_query,
            }

        records = [dict(row) for row in result.rows]
        may_have_more = _may_have_more(result, len(records))
        logger.info(
            "Ontology %s SQL ok in %d ms: %d rows returned, row_count %d%s",
            self._ontology_id,
            int((time.perf_counter() - started) * 1000),
            len(records),
            result.row_count,
            " (page full, may have more)" if may_have_more else "",
        )
        out: dict[str, Any] = {
            "records": records,
            "total_count": len(records),
            "row_count": result.row_count,
            "sql_query": sql_query,
        }
        if may_have_more:
            out["may_have_more"] = True
            out["note"] = (
                "This page is full, so more rows may match. If the question needs "
                "them, read the next page with ORDER BY and OFFSET."
            )
        return out


def _may_have_more(result: Any, returned: int) -> bool:
    """The service's own ``hasMore`` when it sends one, else a full page."""
    reported = getattr(result, "has_more", None)
    if reported is None:
        reported = getattr(result, "hasMore", None)
    if reported is not None:
        return bool(reported)
    return returned in SERVICE_PAGE_SIZES


def create_service_execute_sql_tool(
    client: OntologyServiceClient, ontology_id: str, label: str
) -> BaseTool:
    """The inner ``execute_sql`` tool bound to one ontology on the service."""
    return DataFabricQueryTool(
        name="execute_sql",
        description=(
            f"Execute a SQL SELECT statement over the business objects of the "
            f"ontology '{label}'. Use the object and property names from the "
            "ontology definition in the system message; the runtime maps them to "
            "the data. Retry with a corrected statement when the runtime rejects one."
        ),
        args_schema=DataFabricExecuteSqlInput,
        coroutine=OntologyServiceQueryExecutor(client, ontology_id, label),
        metadata={"tool_type": "ontology_service_sql"},
    )
