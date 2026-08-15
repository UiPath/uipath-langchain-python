"""Just-in-time (JIT) escalation app resolution for Action Center integration.

An escalation may target an app that has not been deployed yet — an *inline*
app that lives only inside the running agent's solution. During a debug run the
app project (``designId``), its ``app_type`` and, for low-code apps, its action
schema must be resolved from the Studio backend at runtime so the human task can
be created without a deployed app.

This module holds that resolution so both the escalation *tool* (factory-path,
``escalation_tool.py``) and the escalate *guardrail action*
(``guardrails/actions/escalate_action.py``) can share one implementation.
"""

import asyncio
import json
import logging
from typing import Any

from uipath.core.feature_flags import FeatureFlags
from uipath.platform import UiPath
from uipath.platform.action_center.tasks import is_low_code_app
from uipath.platform.common import UiPathConfig
from uipath.platform.common._bindings import _resource_overwrites
from uipath.runtime.errors import UiPathErrorCategory

from ..exceptions import AgentRuntimeError, AgentRuntimeErrorCode

_escalation_jit_logger = logging.getLogger(__name__)

_JIT_ESCALATION_APPS_FEATURE_FLAG = "EnableJITEscalationApps"
_IS_DEBUG_RESOLVE_TIMEOUT_SECONDS = 15


def _has_app_name_override(app_name: str | None, folder_path: str | None) -> bool:
    """Return whether a bindings resource overwrite exists for the app.

    Used to disambiguate an ``app_version`` of 1 for backward compatibility: a
    genuinely deployed app carries an app resource binding, whereas an inline
    app (formerly sent from the frontend with a buggy version of 1) does not.

    Mirrors the key resolution used by the ``@resource_override`` decorator on
    ``tasks.create_async`` (``resource_type="app"``,
    ``resource_identifier="app_name"``, ``folder_identifier="app_folder_path"``):
    it looks up ``app.{app_name}`` and prefers the folder-qualified
    ``app.{app_name}.{folder_path}`` when that fuller key is present.

    Args:
        app_name: The escalation channel's design-time app name.
        folder_path: The app folder path, used to disambiguate the overwrite key.

    Returns:
        True when a matching overwrite is present in the current
        :data:`_resource_overwrites` context, False otherwise.
    """
    if not app_name:
        return False

    overwrites = _resource_overwrites.get()
    if not overwrites:
        return False

    key = f"app.{app_name}"
    if folder_path and f"{key}.{folder_path}" in overwrites:
        key = f"{key}.{folder_path}"

    overwrite = overwrites.get(key)
    if overwrite is None:
        return False

    return True


def _is_inline_app(
    app_type: str | None,
    app_version: Any,
    app_name: str | None,
    folder_path: str | None,
) -> bool:
    """Return True when the escalation targets an inline (not-yet-deployed) app.

    An inline app is identified by ``app_version == 0``. Backward compatibility:
    older frontends sent version 1 for inline apps by mistake; a real deployed
    app referenced with version 1 carries an app resource binding, so when no
    binding override exists a version-1 app is treated as inline too.
    """
    if not (is_low_code_app(app_type) or app_type is None):
        return False
    if app_version == 0:
        return True
    return app_version == 1 and not _has_app_name_override(app_name, folder_path)


def _app_type_from_project_type(project_type: str | None) -> str | None:
    """Map a solution project's ``projectType`` to a task ``app_type``.

    The Solution backend reports ``AppV2`` for a coded app and ``Process`` for a
    low code app, whereas task creation expects ``Coded`` / ``Custom``. Returns
    None for an unrecognized or missing ``projectType``.
    """
    return {"AppV2": "Coded", "Process": "Custom"}.get(project_type or "")


async def _resolve_is_debug_run() -> bool:
    """Determine whether the current run is a debug run.

    Reads the running job key (``UIPATH_JOB_KEY``) from the runtime environment
    via ``UiPathConfig``, then retrieves the job from Orchestrator (the SDK
    builds the ``Jobs/...GetByKey`` URL and injects the ``x-uipath-folderkey``
    header from ``UIPATH_FOLDER_KEY`` on every request from its HTTP client).

    The job's ``ParentContext`` field is a JSON string such as
    ``{"IsDebug": true}``; when ``IsDebug`` is truthy the run is a debug run.

    On a successful resolution to a debug run, records it on
    ``UiPathConfig.is_rooted_to_debug_job`` so downstream task creation picks it
    up.

    Returns:
        True when the current job is a debug run, False otherwise (including
        when the job key is unavailable or the parent context cannot be read).
    """
    job_key = UiPathConfig.job_key
    if not job_key:
        return False

    client = UiPath()
    async with asyncio.timeout(_IS_DEBUG_RESOLVE_TIMEOUT_SECONDS):
        job = await client.jobs.retrieve_async(job_key=job_key)

    parent_context_raw = job.parent_context
    if not parent_context_raw:
        return False

    try:
        parent_context = json.loads(parent_context_raw)
    except json.JSONDecodeError:
        _escalation_jit_logger.warning(
            "Unable to parse job ParentContext to determine debug run: %r",
            parent_context_raw,
        )
        return False

    is_debug = bool(parent_context.get("IsDebug"))
    if is_debug:
        UiPathConfig.is_rooted_to_debug_job = True
    return is_debug


async def _resolve_solution_id(client: UiPath) -> str | None:
    """Return the current solution id, resolved lazily from the project id.

    Prefers the value cached on ``UiPathConfig`` (populated when the debug
    runtime applies resource overwrites before the agent runs). Falls back to
    querying the Studio project endpoint directly so escalation still works when
    no resource overwrites were loaded.

    Returns:
        The solution id, or None when it cannot be resolved.
    """
    solution_id = UiPathConfig.studio_solution_id
    if solution_id:
        return solution_id

    project_id = UiPathConfig.project_id
    if not project_id:
        return None

    response = await client.api_client.request_async(
        "GET",
        url=f"/studio_/backend/api/Project/{project_id}",
        scoped="org",
    )
    solution_id = response.json().get("solutionId")
    UiPathConfig.studio_solution_id = solution_id
    return solution_id


async def _resolve_app_project(
    client: UiPath, app_name: str | None
) -> dict[str, Any] | None:
    """Resolve a not-yet-deployed app's project from the solution at runtime.

    Looks up the solution the running agent belongs to and returns the app
    project (``isApp`` true or ``projectType`` is AppV2) whose ``name`` matches ``app_name``.

    Args:
        client: The UiPath SDK client used to call the Studio backend.
        app_name: The escalation channel's app name, used to disambiguate when a
            solution contains more than one app.

    Returns:
        The app project dict, or None when no
        app can be resolved.
    """
    solution_id = await _resolve_solution_id(client)
    if not solution_id:
        return None

    response = await client.api_client.request_async(
        "GET",
        url=f"/studio_/backend/api/Solution/{solution_id}",
        scoped="org",
    )
    apps = [
        project
        for project in response.json().get("projects", [])
        if project.get("isApp") or project.get("projectType") == "AppV2"
    ]

    if app_name is not None:
        for app in apps:
            if app.get("name") == app_name:
                return app

    return None


def _find_schema_file_id(node: Any) -> str | None:
    """Depth-first search the FileOperations structure for the action schema file.

    The app project stores its action schema as ``schemas/schema-<id>.json``.
    Returns the content ``id`` of the first file whose name matches that pattern.
    """
    if not isinstance(node, dict):
        return None

    for file in node.get("files", []):
        name = file.get("name", "")
        if name.startswith("schema-") and name.endswith(".json"):
            return file.get("id")

    for folder in node.get("folders", []):
        found = _find_schema_file_id(folder)
        if found is not None:
            return found

    return None


async def _resolve_app_action_schema(
    client: UiPath, app_project_id: str
) -> dict[str, Any] | None:
    """Fetch a not-yet-deployed app's action schema from the Studio backend.

    Walks the app project's file structure to locate ``schemas/schema-*.json``,
    then reads that file. The returned object mirrors the ``actionSchema`` the
    deployed-apps path returns (``key``, ``inOuts``, ``inputs``, ``outputs``,
    ``outcomes``), so the JIT (debug) task can be created without a deployed app.

    Args:
        client: The UiPath SDK client used to call the Studio backend.
        app_project_id: The app project's ``id`` (not its ``designId``).

    Returns:
        The action schema dict, or None when the schema file cannot be found.
    """
    structure = (
        await client.api_client.request_async(
            "GET",
            url=f"/studio_/backend/api/Project/{app_project_id}/FileOperations/Structure",
            scoped="org",
        )
    ).json()

    file_id = _find_schema_file_id(structure)
    if not file_id:
        return None

    return (
        await client.api_client.request_async(
            "GET",
            url=f"/studio_/backend/api/Project/{app_project_id}/FileOperations/File/{file_id}",
            scoped="org",
        )
    ).json()


async def resolve_is_debug_run_safely() -> bool:
    """Return whether the current run is a debug run, tolerating failures.

    Prefers the flag already recorded on ``UiPathConfig.is_rooted_to_debug_job``
    and otherwise probes Orchestrator, falling back to release mode on any error.
    Mirrors the unconditional debug resolution performed before an escalation
    task is created (a successful probe also records the flag on
    ``UiPathConfig`` for downstream task creation).

    Returns:
        True when the current job is a debug run, False otherwise.
    """
    is_debug = UiPathConfig.is_rooted_to_debug_job
    if not is_debug:
        try:
            is_debug = await _resolve_is_debug_run()
        except Exception:
            # fallback to release mode
            is_debug = False
    return is_debug


async def resolve_jit_escalation_app(
    *,
    app_name: str | None,
    app_version: Any,
    app_type: str | None,
    action_schema: Any,
    folder_path: str | None,
    is_debug: bool,
) -> tuple[str | None, str | None, Any]:
    """Resolve inline (JIT) app targeting info for an escalation task.

    When the escalation targets an inline (not-yet-deployed) app in a debug run
    and the JIT feature flag is enabled, resolves the app project (``designId``),
    its ``app_type`` and — for low-code apps — its action schema from the
    solution at runtime so the human task can be created without a deployed app.

    Args:
        app_name: The design-time app name of the escalation target.
        app_version: The escalation app version (0, or 1 for legacy inline apps).
        app_type: The app type when already known (``Coded`` / ``Custom``), else
            None to resolve from the project's ``projectType``.
        action_schema: The action schema when already known, else None to resolve
            from the Studio backend for low-code apps.
        folder_path: The app folder path, used to disambiguate the overwrite key.
        is_debug: Whether the current run is a debug run.

    Returns:
        The tuple ``(app_project_key, app_type, action_schema)``.
        ``app_project_key`` is None unless the escalation targets an inline app
        that was resolved from the solution at runtime; ``app_type`` and
        ``action_schema`` are returned unchanged when no resolution applies.

    Raises:
        AgentRuntimeError: When a low-code inline app is targeted in debug mode
            but its project key or action schema could not be resolved.
    """
    jit_enabled = FeatureFlags.is_flag_enabled(
        _JIT_ESCALATION_APPS_FEATURE_FLAG, default=False
    )
    is_inline_jit_app = jit_enabled and _is_inline_app(
        app_type, app_version, app_name, folder_path
    )
    if not (is_inline_jit_app and is_debug):
        return None, app_type, action_schema

    app_project_key: str | None = None
    try:
        app_project = await _resolve_app_project(UiPath(), app_name)
        if app_project is not None:
            app_project_key = app_project.get("designId")
            if not app_type:
                app_type = _app_type_from_project_type(app_project.get("projectType"))
            if (
                is_low_code_app(app_type)
                and action_schema is None
                and app_project.get("id")
            ):
                action_schema = await _resolve_app_action_schema(
                    UiPath(), app_project["id"]
                )
    except Exception:
        _escalation_jit_logger.exception(
            "Failed to resolve inline app '%s' from the solution at debug runtime",
            app_name,
        )

    missing_fields = [
        label
        for label, value in (
            ("app project key", app_project_key),
            ("action schema", action_schema),
        )
        if value is None
    ]
    if is_low_code_app(app_type) and missing_fields:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.ESCALATION_APP_JIT_DEBUG_MISSING_INFORMATION,
            title="Unable to create the escalation in debug mode",
            detail=(
                f"Could not resolve the {', '.join(missing_fields)} "
                f"for the app '{app_name}' from the solution at runtime, so the "
                "app cannot be targeted in debug mode. Please open the agent "
                "project and try again"
            ),
            category=UiPathErrorCategory.USER,
        )

    return app_project_key, app_type, action_schema
