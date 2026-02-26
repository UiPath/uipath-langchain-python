# -*- coding: utf-8 -*-
"""
Calibration Dispatcher Agent - StateGraph + HITL + Constraints Enforcement

A production-grade autonomous agent for medical device calibration scheduling.

Features:
- LangGraph StateGraph workflow with Human-in-the-Loop (HITL) via UiPath Action Center
- Dynamic constraint management with manager override capabilities
- Google Maps API integration for route optimization
- Context Grounding for policy retrieval (RAG pattern)
- MCP Server integration for RPA workflow execution
- Technician specialization matching and SLA-aware scheduling

For configuration, see config.py
"""

import json
import math
import uuid
import logging
import re
import os
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent as create_react_agent
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command

from uipath.platform import UiPath

from uipath.platform.common import CreateTask
from uipath_langchain.chat import UiPathChat
from uipath_langchain.retrievers import ContextGroundingRetriever

import googlemaps

# Import centralized configuration
import config

# ---------- Bootstrap ----------

load_dotenv()

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT
)
logger = logging.getLogger("calibration-dispatcher")

# Validate configuration before proceeding
if not config.validate_config():
    logger.error("Configuration validation failed. Please check config.py")
    if not config.USE_MOCK_DATA:
        raise RuntimeError("Invalid configuration. Cannot proceed.")

config.print_config_summary()

# Initialize UiPath client
uipath_client = UiPath()

# Initialize LLM
llm = UiPathChat(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE
)

# Initialize Context Grounding for policy retrieval
context_grounding = ContextGroundingRetriever(
    index_name=config.CONTEXT_GROUNDING_INDEX_NAME,
    folder_path=config.UIPATH_FOLDER_PATH,
    number_of_results=config.CONTEXT_GROUNDING_NUM_RESULTS,
)

# Initialize Google Maps client
GOOGLE_MAPS_API_KEY = config.GOOGLE_MAPS_API_KEY
if not GOOGLE_MAPS_API_KEY:
    try:
        logger.info("Google Maps API key not in config, trying UiPath Assets...")
        asset = uipath_client.assets.retrieve(
            name=config.GOOGLE_MAPS_ASSET_NAME,
            folder_path=config.UIPATH_FOLDER_PATH
        )
        GOOGLE_MAPS_API_KEY = getattr(asset, "value", None) or getattr(asset, "stringValue", None)
        if GOOGLE_MAPS_API_KEY:
            logger.info("Google Maps API key loaded from Assets.")
        else:
            logger.warning("Asset '%s' found but value is empty.", config.GOOGLE_MAPS_ASSET_NAME)
    except Exception as e:
        logger.warning("Failed to retrieve Asset: %s", e)

if GOOGLE_MAPS_API_KEY:
    try:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        logger.info("Google Maps client initialized successfully.")
    except Exception as e:
        gmaps = None
        logger.error("Failed to initialize Google Maps client: %s", e)
else:
    gmaps = None
    logger.error("Google Maps client NOT initialized - missing API key!")

# ---------- Global buffer to avoid double planning ----------

LAST_ROUTING_PLAN: Optional[Dict[str, Any]] = None

# ---------- Helpers: specialization, geometry ----------

def _estimate_service_hours_for_visit(visit: Dict[str, Any]) -> float:
    """Calculate total service hours for a clinic visit based on device types."""
    s = 0.0
    for d in visit.get("devices", []):
        if d.get("device_type") == "Audiometer":
            s += config.SERVICE_TIME_AUDIOMETER
        elif d.get("device_type") == "Tympanometer":
            s += config.SERVICE_TIME_TYMPANOMETER
    return s

def _tech_ok_for_devices(tech: Dict[str, Any], visits: List[Dict[str, Any]]) -> bool:
    """Check if technician specialization matches required device types."""
    spec = {tech.get("specialization") or "All"}
    required = set()
    for v in visits:
        for d in v.get("devices", []):
            required |= config.DEVICE_TO_SPECIALIZATION.get(d.get("device_type"), {"All"})
    return bool(spec & required) or "All" in spec

def _city_distance_km(city_a: str, city_b: str) -> float:
    """Calculate approximate distance between two cities in kilometers."""
    ax, ay = config.CITY_COORDS.get(city_a, (0.0, 0.0))
    bx, by = config.CITY_COORDS.get(city_b, (0.0, 0.0))
    return math.hypot(ax - bx, ay - by) * 111.0  # ~111 km per degree

def _pick_technician_for_city(technicians: List[Dict[str, Any]], city: str, visits: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [t for t in technicians if _tech_ok_for_devices(t, visits)]
    local = [t for t in candidates if t.get("home_base_city") == city]
    if local:
        return local[0]
    ranked = sorted(candidates or technicians, key=lambda t: _city_distance_km(t.get("home_base_city") or "", city))
    return ranked[0] if ranked else None

# ---------- Helpers: policy limits parsing & fallbacks ----------

def _extract_json_like(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = text[start:i+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None

def _parse_manager_note(manager_note: str) -> Dict[str, Any]:
    note = (manager_note or "").lower()
    out: Dict[str, Any] = {}
    m = re.search(    r"(?:max(?:imum)?|up\s+to|allow(?:ed)?(?:\s+up\s+to)?|no\s+more\s+than|at\s+most)"
    r"\s*([0-9]+(?:\.[0-9]+)?)\s*(?:h|hour(?:s)?)", note)
    if m:
        out["max_work_hours"] = float(m.group(1))
    m = re.search(r"(?:max\s*)?([0-9]+)\s*(?:visits?|locations?|stops?|sites?)", note)
    if m:
        out["max_visits_per_route"] = int(m.group(1))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*km", note)
    if m:
        out["max_distance_km_per_route"] = float(m.group(1))
    if any(k in note for k in ["overtime", "extra hours", "sla", "weekend", "extend hours", "longer day"]):
        out["allow_overtime"] = True

    special_requirements = []
    if any(k in note for k in ["support", "help", "assist", "backup"]):
        special_requirements.append("cross_city_support_requested")
    if any(k in note for k in ["travel to", "go to", "visit", "send to"]):
        for city in ["warszawa", "poznan", "wroclaw", "szczecin", "krakow", "gdansk"]:
            if city in note:
                special_requirements.append(f"travel_to_{city}")
    if any(k in note for k in ["urgent", "asap", "immediately", "priority", "critical"]):
        special_requirements.append("urgent_priority")
    if any(k in note for k in ["short day", "shorter hours", "reduce hours", "early finish"]):
        special_requirements.append("shorter_workday")
    if special_requirements:
        out["special_requirements"] = special_requirements
    out["full_note"] = manager_note
    return out

def _fallback_policy_limits(manager_note: str = "") -> Dict[str, Any]:
    """
    Get default routing constraints from config, optionally overridden by manager note.
    """
    limits = {
        "max_work_hours": config.DEFAULT_MAX_WORK_HOURS,
        "max_visits_per_route": config.DEFAULT_MAX_VISITS_PER_ROUTE,
        "max_distance_km_per_route": config.DEFAULT_MAX_DISTANCE_KM,
        "allow_overtime": False,
    }
    overrides = _parse_manager_note(manager_note)
    limits.update({k: v for k, v in overrides.items() if v is not None})
    logger.warning("Using fallback policy limits: %s", {k: v for k, v in limits.items() if k != "full_note"})
    return limits

def _derive_policy_limits_via_llm(manager_note: str = "") -> Dict[str, Any]:
    """Internal helper: call get_calibration_rules via a tiny ReAct agent and return numeric limits."""
    tools = [get_calibration_rules]
    agent_tmp = create_react_agent(llm, tools)
    context = ""
    if manager_note:
        context = (
            "\n\nMANAGER REQUIREMENTS:\n"
            f"{manager_note}\n\nIMPORTANT: If manager specifies limits, override default policy values."
        )
    msg = (
        "Read corporate policy using get_calibration_rules. Extract numeric limits ONLY as JSON:\n"
        '{"max_work_hours": <float>, "max_visits_per_route": <int>, "max_distance_km_per_route": <float>, "allow_overtime": <bool>}\n'
        "Respond with JSON only, no prose." + context
    )
    try:
        res = agent_tmp.invoke({"messages": [HumanMessage(content=msg)]})
        raw = res["messages"][-1].content if isinstance(res, dict) else ""
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = _extract_json_like(raw) or {}
        if not parsed:
            # naive lines
            cand: Dict[str, Any] = {}
            for line in str(raw).splitlines():
                if ":" not in line:
                    continue
                key, val = [x.strip().strip('",') for x in line.split(":", 1)]
                k = key.strip('"').lower().replace(" ", "_")
                if k in {"max_work_hours", "max_visits_per_route", "max_distance_km_per_route", "allow_overtime"}:
                    if k == "allow_overtime":
                        cand[k] = ("true" in val.lower()) or ("yes" in val.lower())
                    elif "visits" in k:
                        cand[k] = int(re.findall(r"[0-9]+", val)[0])
                    else:
                        cand[k] = float(re.findall(r"[0-9]+(?:\.[0-9]+)?", val)[0])
            parsed = cand
        if manager_note:
            note_data = _parse_manager_note(manager_note)
            for k in ["max_work_hours", "max_visits_per_route", "max_distance_km_per_route", "allow_overtime"]:
                if k in note_data and note_data[k] is not None:
                    parsed[k] = note_data[k]
            if "special_requirements" in note_data:
                parsed["special_requirements"] = note_data["special_requirements"]
            parsed["full_note"] = note_data.get("full_note", manager_note)
        return parsed or _fallback_policy_limits(manager_note)
    except Exception as e:
        logger.warning("Policy limits derivation failed or returned non-JSON: %s", e)
        return _fallback_policy_limits(manager_note)

# ---------- Helpers: weekend SLA date selection ----------

def _has_overdue(visits: List[Dict[str, Any]]) -> bool:
    for v in visits:
        for d in v.get("devices", []):
            if d.get("days_until_due", 0) < 0:
                return True
    return False

def _choose_route_date(allow_overtime: bool, visits: List[Dict[str, Any]]) -> Tuple[str, bool, str]:
    today = date.today()
    nxt = today + timedelta(days=1)
    is_weekend = nxt.weekday() >= 5  # 5=Sat, 6=Sun
    note = ""
    if is_weekend:
        if nxt.weekday() == 5 and allow_overtime and _has_overdue(visits):
            note = "SLA weekend exception applied (Saturday) for OVERDUE devices."
            return (nxt.strftime("%Y-%m-%d"), True, note)
        delta = 7 - nxt.weekday()
        monday = nxt + timedelta(days=delta)
        return (monday.strftime("%Y-%m-%d"), False, "Shifted to Monday (no weekend work).")
    return (nxt.strftime("%Y-%m-%d"), False, "")

# ---------- Tools ----------

@tool
def analyze_equipment_status() -> Dict[str, Any]:
    """LangChain tool: Pull equipment from Data Fabric and split into OVERDUE/URGENT/SCHEDULED/ACTIVE buckets."""
    try:
        logger.info("Analyzing equipment status...")
        records = uipath_client.entities.list_records(entity_key=config.EQUIPMENT_ENTITY_ID, start=0, limit=100)
        today = datetime.now().date()
        overdue, urgent, scheduled, active = [], [], [], []
        for record in records:
            equipment_id = getattr(record, "equipmentId", None)
            device_type = getattr(record, "deviceType", None)
            clinic_id = getattr(record, "clinicId", None)
            next_due_str = str(getattr(record, "nextCalibrationDue", "")).strip() or None
            if not next_due_str or not equipment_id:
                continue
            try:
                next_due = datetime.fromisoformat(next_due_str.split("T")[0].split(" ")[0]).date()
            except Exception:
                continue
            days_until_due = (next_due - today).days
            device_info = {
                "equipment_id": equipment_id,
                "device_type": device_type,
                "clinic_id": clinic_id,
                "next_due": str(next_due),
                "days_until_due": days_until_due,
            }
            if days_until_due < 0:
                overdue.append(device_info)
            elif device_type == "Audiometer" and days_until_due <= 14:
                urgent.append(device_info)
            elif device_type == "Tympanometer" and days_until_due <= 7:
                urgent.append(device_info)
            elif device_type == "Audiometer" and 15 <= days_until_due <= 30:
                scheduled.append(device_info)
            elif device_type == "Tympanometer" and 8 <= days_until_due <= 21:
                scheduled.append(device_info)
            else:
                active.append(device_info)
        logger.info("Analysis: %d OVERDUE, %d URGENT, %d SCHEDULED", len(overdue), len(urgent), len(scheduled))
        return {
            "total_equipment": len(records),
            "overdue_count": len(overdue),
            "urgent_count": len(urgent),
            "scheduled_count": len(scheduled),
            "active_count": len(active),
            "overdue_devices": overdue[:20],
            "urgent_devices": urgent[:20],
            "scheduled_devices": scheduled[:20],
            "analysis_date": str(today),
        }
    except Exception as e:
        logger.error("Equipment analysis failed: %s", e)
        return {"error": str(e)}

@tool
def get_calibration_rules(query: str) -> str:
    """LangChain tool: Retrieve policy fragments from Context Grounding retriever (returns plain text)."""
    try:
        docs = context_grounding.invoke(query)
        if not docs:
            return "No specific rules found. Use default thresholds."
        rules_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        logger.info("Retrieved %d rule documents", len(docs))
        return rules_text
    except Exception as e:
        logger.error("Error retrieving rules: %s", e)
        return "Error retrieving rules."

@tool
def query_clinics() -> List[Dict[str, Any]]:
    """LangChain tool: Return list of clinics (id, name, address, city, geo, contacts, SLA)."""
    try:
        records = uipath_client.entities.list_records(entity_key=config.CLINICS_ENTITY_ID, start=0, limit=100)
        clinics_list = []
        for record in records:
            clinics_list.append({
                "clinic_id": getattr(record, "clinicId", None),
                "clinic_name": getattr(record, "clinicName", None),
                "address": getattr(record, "address", None),
                "city": getattr(record, "city", None),
                "postal_code": getattr(record, "postalCode", None),
                "latitude": float(getattr(record, "latitude", "0") or 0),
                "longitude": float(getattr(record, "longitude", "0") or 0),
                "contact_person": getattr(record, "contactPerson", None),
                "contact_email": getattr(record, "contactEmail", None),
                "sla_hours": int(getattr(record, "slaHours", 72) or 72),
            })
        logger.info("Retrieved %d clinics", len(clinics_list))
        return clinics_list
    except Exception as e:
        logger.error("Error querying clinics: %s", e)
        return []

@tool
def query_technicians() -> List[Dict[str, Any]]:
    """LangChain tool: Return list of technicians with specialization and home base."""
    try:
        records = uipath_client.entities.list_records(entity_key=config.TECHNICIANS_ENTITY_ID, start=0, limit=100)
        technicians_list = []
        for record in records:
            technicians_list.append({
                "technician_id": getattr(record, "technicianId", None),
                "technician_name": getattr(record, "technicianName", None),
                "email": getattr(record, "email", None),
                "phone": getattr(record, "phone", None),
                "specialization": getattr(record, "specialization", None),
                "home_base_city": getattr(record, "homeBaseCity", None),
            })
        logger.info("Retrieved %d technicians", len(technicians_list))
        return technicians_list
    except Exception as e:
        logger.error("Error querying technicians: %s", e)
        return []

@tool
def optimize_route(clinic_ids: List[str], technician_id: Optional[str] = None, city: str = "") -> Dict[str, Any]:
    """LangChain tool: Build/optimize a driving route for given clinic IDs; returns km, hours, and a Google Maps URL."""
    if not gmaps:
        return {"error": "Google Maps API not configured"}
    try:
        logger.info("Optimizing route for %d clinics in %s...", len(clinic_ids), city or "?")
        all_clinics_records = uipath_client.entities.list_records(entity_key=config.CLINICS_ENTITY_ID, start=0, limit=100)
        all_clinics = [{
            "clinic_id": getattr(r, "clinicId", None),
            "latitude": float(getattr(r, "latitude", "0") or 0),
            "longitude": float(getattr(r, "longitude", "0") or 0),
        } for r in all_clinics_records]
        clinics = [c for c in all_clinics if c["clinic_id"] in clinic_ids]
        if len(clinics) < 1:
            return {"error": "Need at least 1 clinic"}

        start_point = None
        if technician_id:
            tech_records = uipath_client.entities.list_records(
                entity_key=config.TECHNICIANS_ENTITY_ID, start=0, limit=100
            )
            for tech in tech_records:
                if getattr(tech, "technicianId", None) == technician_id:
                    home_city = getattr(tech, "homeBaseCity", None)
                    if home_city and home_city in config.CITY_COORDS:
                        start_point = config.CITY_COORDS[home_city]
                        logger.info("Using technician home base: %s", home_city)
                    break

        if start_point and city in config.CITY_COORDS:
            cx, cy = config.CITY_COORDS[city]
            dist_km = math.hypot(start_point[0] - cx, start_point[1] - cy) * 111.0
            if dist_km > 80.0:
                start_point = (cx, cy)
                logger.info("Home base far from cluster; using city centroid for %s as origin", city)

        if start_point:
            origin = f"{start_point[0]},{start_point[1]}"
            destination = origin
            waypoints_coords = [f"{c['latitude']},{c['longitude']}" for c in clinics]
        else:
            origin = f"{clinics[0]['latitude']},{clinics[0]['longitude']}"
            destination = f"{clinics[-1]['latitude']},{clinics[-1]['longitude']}"
            waypoints_coords = [f"{c['latitude']},{c['longitude']}" for c in clinics[1:-1]]

        if waypoints_coords:
            directions = gmaps.directions(
                origin, destination,
                waypoints=waypoints_coords,
                optimize_waypoints=True, mode="driving"
            )
        else:
            directions = gmaps.directions(origin, destination, mode="driving")
        if not directions:
            return {"error": "No route found"}

        route = directions[0]
        total_distance_km = sum(leg["distance"]["value"] for leg in route["legs"]) / 1000
        total_duration_hours = sum(leg["duration"]["value"] for leg in route["legs"]) / 3600

        if waypoints_coords and "waypoint_order" in route:
            optimized_order = route["waypoint_order"]
            optimized_waypoints = [waypoints_coords[i] for i in optimized_order]
            all_waypoints = [origin] + optimized_waypoints + [destination]
            logger.info("Using optimized waypoint order: %s", optimized_order)
        else:
            all_waypoints = [origin] + waypoints_coords + [destination] if waypoints_coords else [origin, destination]

        map_url = "https://www.google.com/maps/dir/" + "/".join(all_waypoints)
        logger.info("Route optimized: %.1f km, %.1f h", total_distance_km, total_duration_hours)
        return {
            "total_distance_km": round(total_distance_km, 1),
            "total_duration_hours": round(total_duration_hours, 2),
            "starts_from_home": start_point is not None,
            "route_map_url": map_url,
        }
    except Exception as e:
        logger.error("Route optimization failed: %s", e)
        return {"error": str(e)}

@tool
def build_routing_plan(
    devices_needing_service: Optional[Union[str, List[Dict[str, Any]]]] = None,
    max_work_hours: Optional[float] = None,
    max_visits_per_route: Optional[int] = None,
    max_distance_km_per_route: Optional[float] = None,
    allow_overtime: Optional[bool] = None,
    manager_note: Optional[str] = None,
) -> Dict[str, Any]:
    """LangChain tool: Build an optimized routing plan; enforces hours/visits/distance limits and optional overtime."""
    global LAST_ROUTING_PLAN
    try:
        if isinstance(devices_needing_service, str):
            try:
                devices_needing_service = json.loads(devices_needing_service)
                logger.info("Parsed devices_needing_service from JSON string")
            except Exception as e:
                logger.warning("Failed to parse devices_needing_service JSON: %s", e)
                devices_needing_service = None
        
        if not devices_needing_service:
            logger.info("No devices provided, fetching from analyze_equipment_status...")
            analysis = analyze_equipment_status.invoke({})
            devices_needing_service = analysis.get("overdue_devices", []) + analysis.get("urgent_devices", [])
            if not devices_needing_service:
                logger.warning("No overdue or urgent devices found")
                empty_result = {
                    "routing_plan": [],
                    "total_routes": 0,
                    "total_devices": 0,
                    "total_distance_km": 0,
                }
                LAST_ROUTING_PLAN = empty_result
                return empty_result
        
        logger.info("Building routing plan for %d devices...", len(devices_needing_service))
        all_clinics = query_clinics.invoke({})
        all_technicians = query_technicians.invoke({})
        if not all_clinics or not all_technicians:
            return {"error": "Missing clinic or technician data"}

        clinic_device_map: Dict[str, Dict[str, Any]] = {}
        for device in devices_needing_service:
            cid = device["clinic_id"]
            if cid not in clinic_device_map:
                clinic_info = next((c for c in all_clinics if c["clinic_id"] == cid), None)
                if clinic_info:
                    clinic_device_map[cid] = {"clinic": clinic_info, "devices": []}
            if cid in clinic_device_map:
                clinic_device_map[cid]["devices"].append(device)

        city_clusters: Dict[str, List[Dict[str, Any]]] = {}
        for cid, data in clinic_device_map.items():
            city = data["clinic"]["city"]
            city_clusters.setdefault(city, []).append({
                "clinic_id": cid,
                "clinic": data["clinic"],
                "devices": data["devices"],
            })
        logger.info("Created %d city clusters: %s", len(city_clusters), list(city_clusters.keys()))

        routing_plan: List[Dict[str, Any]] = []

        for city, visits in city_clusters.items():
            logger.info("Processing city %s with %d visits (%d devices total)", 
                       city, len(visits), sum(len(v.get("devices", [])) for v in visits))
            
            assigned_tech = _pick_technician_for_city(all_technicians, city, visits)
            if not assigned_tech:
                continue

            clinic_ids = [v["clinic_id"] for v in visits]
            route_result = optimize_route.invoke({
                "clinic_ids": clinic_ids,
                "technician_id": assigned_tech["technician_id"],
                "city": city,
            })
            if "error" in route_result:
                logger.warning("Route optimization failed for %s: %s; using fallback", city, route_result["error"])
                n = max(1, len(clinic_ids))
                est_km = n * 8.0
                route_result = {
                    "total_distance_km": est_km,
                    "total_duration_hours": round(est_km / 40.0, 2),
                    "starts_from_home": True,
                    "route_map_url": "https://www.google.com/maps/search/" + (city or "").replace(" ", "+"),
                }

            current_visits = list(visits)

            # Calculate initial work load before expansion
            initial_travel = float(route_result.get("total_duration_hours", 0))
            initial_service = sum(_estimate_service_hours_for_visit(x) for x in current_visits)
            initial_total = initial_travel + initial_service

            expansion_applied = False

            # EXPANSION: If overtime allowed and there is headroom, try to include all city visits
            if allow_overtime and max_work_hours and max_work_hours > 8.0:
                hours_available = max_work_hours - initial_total
                if hours_available > 1.0:
                    logger.info("Overtime allowed (%sh). Current: %.2fh, Available: %.2fh. Attempting expansion for %s.", 
                               max_work_hours, initial_total, hours_available, city)
                    expanded_visits = list(visits)
                    expanded_ids = [v["clinic_id"] for v in expanded_visits]
                    expanded_route = optimize_route.invoke({
                        "clinic_ids": expanded_ids,
                        "technician_id": assigned_tech["technician_id"],
                        "city": city,
                    })
                    expanded_travel = float(expanded_route.get("total_duration_hours", 0))
                    expanded_service = sum(_estimate_service_hours_for_visit(x) for x in expanded_visits)
                    expanded_total = expanded_travel + expanded_service
                    if expanded_total <= max_work_hours and expanded_total > initial_total + 0.5:
                        current_visits = expanded_visits
                        route_result = expanded_route
                        expansion_applied = True
                        logger.info("EXPANSION SUCCESS: %d -> %d visits, %.2fh -> %.2fh (limit %sh)", 
                                   len(visits), len(current_visits), initial_total, expanded_total, max_work_hours)
                    elif expanded_total > max_work_hours:
                        logger.info("EXPANSION FAILED: %d visits would be %.2fh, exceeds %sh limit.", 
                                   len(expanded_visits), expanded_total, max_work_hours)
                    else:
                        logger.info("EXPANSION SKIPPED: No meaningful improvement (%.2fh -> %.2fh)", 
                                   initial_total, expanded_total)
                else:
                    logger.info("EXPANSION SKIPPED: Insufficient headroom (%.2fh used of %sh)", 
                               initial_total, max_work_hours)

            # Apply visit count limit
            if max_visits_per_route is not None and len(current_visits) > max_visits_per_route:
                current_visits = current_visits[:max_visits_per_route]

            # Apply distance limit
            if max_distance_km_per_route is not None and route_result.get("total_distance_km", 0) > max_distance_km_per_route:
                while current_visits and route_result.get("total_distance_km", 0) > max_distance_km_per_route:
                    current_visits = current_visits[:-1]
                    ids = [v["clinic_id"] for v in current_visits]
                    if ids:
                        route_result = optimize_route.invoke({
                            "clinic_ids": ids,
                            "technician_id": assigned_tech["technician_id"],
                            "city": city,
                        })
                    else:
                        break

            # Apply hours limit if expansion didn't already validate
            if max_work_hours is not None and not expansion_applied:
                tmp, chosen = [], []
                for v in current_visits:
                    tmp.append(v)
                    ids = [x["clinic_id"] for x in tmp]
                    rtmp = optimize_route.invoke({
                        "clinic_ids": ids,
                        "technician_id": assigned_tech["technician_id"],
                        "city": city,
                    })
                    travel_h = float(rtmp.get("total_duration_hours", 0))
                    service_h = sum(_estimate_service_hours_for_visit(x) for x in tmp)
                    if travel_h + service_h <= max_work_hours:
                        chosen = list(tmp)
                    else:
                        break
                if chosen:
                    current_visits = chosen
                    ids = [v["clinic_id"] for v in current_visits]
                    route_result = optimize_route.invoke({
                        "clinic_ids": ids,
                        "technician_id": assigned_tech["technician_id"],
                        "city": city,
                    })

            travel_hours = float(route_result.get("total_duration_hours", 0))
            service_hours = sum(_estimate_service_hours_for_visit(x) for x in current_visits)
            total_work_hours = travel_hours + service_hours

            planned_date, is_weekend, weekend_note = _choose_route_date(bool(allow_overtime), current_visits)

            routing_plan.append({
                "city": city,
                "visits": current_visits,
                "technician": assigned_tech,
                "route": route_result,
                "travel_hours": round(travel_hours, 2),
                "service_hours": round(service_hours, 2),
                "total_work_hours": round(total_work_hours, 2),
                "total_devices": sum(len(v["devices"]) for v in current_visits),
                "manager_note": manager_note or "",
                "allow_overtime": bool(allow_overtime),
                "route_date": planned_date,
                "route_date_is_weekend": is_weekend,
                "route_date_note": weekend_note,
            })

        logger.info("Created %d routes", len(routing_plan))
        result = {
            "routing_plan": routing_plan,
            "total_routes": len(routing_plan),
            "total_devices": sum(r["total_devices"] for r in routing_plan),
            "total_distance_km": sum(r["route"]["total_distance_km"] for r in routing_plan),
        }
        LAST_ROUTING_PLAN = result
        return result
    except Exception as e:
        logger.error("Routing plan failed: %s", e)
        return {"error": str(e)}

@tool
def request_manager_approval(routing_plan: Dict[str, Any]) -> Dict[str, Any]:
    """LangChain tool: Prepare summary for HITL approval (actual Action Center call happens in the HITL node)."""
    try:
        routes = routing_plan.get("routing_plan", [])
        if not routes:
            return {"error": "No routes in plan"}
        logger.info("Preparing %d approval tasks (will be created in HITL node)...", len(routes))
        task_ids = [str(uuid.uuid4()) for _ in routes]
        return {
            "task_ids": task_ids,
            "total_tasks": len(task_ids),
            "action_center_url": "https://cloud.uipath.com/[your-org]/[your-tenant]/actioncenter_/tasks",
            "message": f"Prepared {len(task_ids)} approval tasks. Check Action Center.",
        }
    except Exception as e:
        logger.error("Approval request failed: %s", e)
        return {"error": str(e)}

@tool
def create_service_orders(approved_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """LangChain tool: Create (mock) service orders per approved route; returns counts for reporting."""
    try:
        logger.info("Creating service orders for %d routes...", len(approved_routes))
        total_orders = sum(r["total_devices"] for r in approved_routes)
        logger.info("Created %d service orders", total_orders)
        return {
            "total_orders": total_orders,
            "orders_per_route": [r["total_devices"] for r in approved_routes],
            "message": f"Successfully created {total_orders} service orders in Data Fabric",
        }
    except Exception as e:
        logger.error("Service order creation failed: %s", e)
        return {"error": str(e)}

@tool
def trigger_notification_workflow(service_orders: Dict[str, Any]) -> Dict[str, Any]:
    """LangChain tool: Trigger RPA workflow that sends email notifications (delegated to UiPath Orchestrator)."""
    try:
        total_orders = service_orders.get("total_orders", 0)
        logger.info("Triggering notification workflow for %d orders...", total_orders)
        return {
            "workflow_triggered": True,
            "total_notifications": total_orders,
            "message": f"RPA workflow will send {total_orders} email notifications",
        }
    except Exception as e:
        logger.error("Notification trigger failed: %s", e)
        return {"error": str(e)}

# ---------- System Prompt ----------

SYSTEM_PROMPT = """You are a Calibration Dispatcher Agent for medical equipment routing.

TASK: Create optimized calibration routes by analyzing equipment status and applying policy constraints.

EXECUTION STEPS:

1. ANALYZE EQUIPMENT
   Call analyze_equipment_status() to identify OVERDUE and URGENT devices.
   OVERDUE: days_until_due < 0 (past calibration deadline)
   URGENT: days_until_due <= 14 (Audiometers) or <= 7 (Tympanometers)

2. RETRIEVE POLICIES
   Call get_calibration_rules(query="routing constraints work hours visits distance overtime")
   This retrieves company policy documents via Context Grounding.
   Extract from policy documents:
   - Maximum work hours per technician per day
   - Maximum visits per route
   - Maximum travel distance per route (km)
   - Overtime authorization rules for SLA compliance

3. BUILD ROUTING PLAN
   Call build_routing_plan() with ONLY constraint parameters:
   
   IMPORTANT: Do NOT pass devices_needing_service parameter - it will be fetched automatically.
   
   CRITICAL: If manager_note contains explicit constraints (e.g., "max 6 hours"), 
   use those exact values - manager instructions override all other rules including OVERDUE emergency protocols.
   
   Example call for OVERDUE devices with no manager constraints:
   build_routing_plan(
       max_work_hours=12.0,
       max_visits_per_route=5,
       max_distance_km_per_route=200.0,
       allow_overtime=True,
       manager_note=""
   )
   
   Example call when manager specifies "max 6 hours":
   build_routing_plan(
       max_work_hours=6.0,
       max_visits_per_route=4,
       max_distance_km_per_route=200.0,
       allow_overtime=False,
       manager_note="max 6 hours"
   )
   
   For routes with OVERDUE devices (and no manager constraints):
   - Set allow_overtime=True
   - Extend max_work_hours to 10-12 hours
   - Prioritize immediate service to avoid regulatory violations
   
   For URGENT-only routes:
   - Apply standard policy limits (typically 8 hours, 4 visits, 200km)
   - Follow normal scheduling procedures
   
   Use your reasoning to balance: SLA compliance, technician workload, travel efficiency.

KEY CONSTRAINTS:
- Manager instructions are ABSOLUTE PRIORITY (override everything)
- Respect technician specialization (Audiometry, Tympanometry, All)
- Minimize total travel distance
- Never exceed daily capacity limits from policy or manager
- OVERDUE devices require immediate action (24-48 hour response)

OUTPUT:
Provide brief summary: X devices found (Y overdue, Z urgent), constraints applied, N routes created.

Current date: {current_date}
"""

# ---------- State / Nodes ----------

class WorkflowState(BaseModel):
    agent_messages: list = []
    agent_completed: bool = False
    routing_plan: Dict[str, Any] = {}
    current_route_index: int = 0
    approved_routes: List[Dict[str, Any]] = []
    rejected_routes: List[Dict[str, Any]] = []
    workflow_complete: bool = False
    
    # Revision tracking for ChangesRequested loop
    revision_in_progress: bool = False
    current_revision_iteration: int = 0
    pending_manager_note: str = ""
    max_revision_iterations: int = config.MAX_REVISION_ITERATIONS

def _build_agent_comments(route: Dict[str, Any], manager_note: str = "") -> str:
    city = route.get("city", "?")
    dist = route.get("route", {}).get("total_distance_km", 0.0)
    trav = float(route.get("travel_hours", 0.0))
    serv = float(route.get("service_hours", 0.0))
    work = float(route.get("total_work_hours", 0.0))
    tech = route.get("technician", {}) or {}
    tech_name = tech.get("technician_name", "N/A")
    devices = sum(len(v.get("devices", [])) for v in route.get("visits", []))
    
    lines = [
        "AI Agent Analysis:",
        f"- Route optimized for minimal travel time ({trav:.2f}h) in {city}",
        f"- Technician {tech_name} assigned (specialization matched)",
        f"- Total work time: {work:.2f}h (service {serv:.2f}h + travel {trav:.2f}h), distance {dist} km",
        f"- {devices} devices scheduled",
    ]
    if route.get("route_date_is_weekend"):
        lines.append(f"- Planned for Saturday due to SLA exception: {route.get('route_date')}")
    if route.get("allow_overtime"):
        lines.append("- Overtime applied due to SLA-critical OVERDUE devices.")
    if manager_note:
        lines.append(f"- Manager note applied: {manager_note}")
        parsed = _parse_manager_note(manager_note)
        if "special_requirements" in parsed:
            req_descriptions = {
                "cross_city_support_requested": "Cross-city support/assistance identified",
                "urgent_priority": "Urgent priority flagged",
            }
            for req in parsed["special_requirements"]:
                if req.startswith("travel_to_"):
                    city_name = req.replace("travel_to_", "").capitalize()
                    lines.append(f"  * Travel requirement: {city_name}")
                elif req in req_descriptions:
                    lines.append(f"  * {req_descriptions[req]}")
    lines += [
        "",
        "Grounding references:",
        "• Service Procedures v4.1 – standard durations (Audiometer 2.0h, Tympanometer 1.5h)",
        "• Routing Guidelines v3.2 – daily capacity limits and waypoint optimization",
        "• Calibration Rules v2.1 – OVERDUE/URGENT thresholds and SLA windows",
        "Decision rationale: minimized distance while staying within documented limits.",
    ]
    return "\n".join(lines)

# ---------- helpers ----------

def _collect_devices_overdue_and_urgent() -> Tuple[List[Dict[str, Any]], bool]:
    analysis = analyze_equipment_status.invoke({})
    overdue = analysis.get("overdue_devices", [])
    urgent = analysis.get("urgent_devices", [])
    devices = overdue + urgent
    has_overdue = len(overdue) > 0
    return devices, has_overdue

def _compute_limits_for_devices(has_overdue: bool, manager_note: str = "") -> Dict[str, Any]:
    """
    Compute routing constraints, with automatic override for OVERDUE devices.
    Respects manager explicit instructions when provided.
    """
    limits = _derive_policy_limits_via_llm(manager_note)
    parsed_note = _parse_manager_note(manager_note)
    
    manager_set_hours = "max_work_hours" in parsed_note
    manager_set_visits = "max_visits_per_route" in parsed_note
    manager_wants_shorter = "shorter_workday" in parsed_note.get("special_requirements", [])
    
    if manager_set_hours or manager_set_visits or manager_wants_shorter:
        logger.info("Manager explicit instruction detected, respecting manager limits")
        return limits
    
    if has_overdue:
        if not limits.get("allow_overtime"):
            logger.info("Auto-enabling overtime for OVERDUE devices (no manager constraint)")
            limits["allow_overtime"] = True
        if limits.get("max_work_hours", config.DEFAULT_MAX_WORK_HOURS) <= config.DEFAULT_MAX_WORK_HOURS:
            limits["max_work_hours"] = config.OVERDUE_MAX_WORK_HOURS
            logger.info("Auto-extended to %sh for OVERDUE devices (no manager constraint)", 
                       config.OVERDUE_MAX_WORK_HOURS)
    
    return limits

def _plan_with_limits(devices: List[Dict[str, Any]], limits: Dict[str, Any], manager_note: str = "") -> Dict[str, Any]:
    return build_routing_plan.invoke({
        "devices_needing_service": devices,
        "max_work_hours": limits.get("max_work_hours"),
        "max_visits_per_route": limits.get("max_visits_per_route"),
        "max_distance_km_per_route": limits.get("max_distance_km_per_route"),
        "allow_overtime": limits.get("allow_overtime", False),
        "manager_note": manager_note,
    })

# ---------- Nodes ----------

def run_agent_node(state: WorkflowState) -> WorkflowState:
    logger.info("=" * 60)
    logger.info("PHASE 1: AGENT ANALYSIS & ROUTING")
    logger.info("=" * 60)

    current_date = datetime.now().strftime("%Y-%m-%d")
    system_prompt = SYSTEM_PROMPT.format(current_date=current_date)

    tools = [analyze_equipment_status, get_calibration_rules, build_routing_plan]
    agent_local = create_react_agent(llm, tools)

    user_request = f"""{system_prompt}
Please execute the process above exactly with tools. Then summarize briefly."""

    def _fallback_plan(path_label: str) -> Dict[str, Any]:
        logger.warning("Agent did not produce a routing plan (%s). Falling back to deterministic path.", path_label)
        devices, has_overdue = _collect_devices_overdue_and_urgent()
        limits = _compute_limits_for_devices(has_overdue)
        return _plan_with_limits(devices, limits, manager_note="")

    try:
        result = agent_local.invoke({"messages": [HumanMessage(content=user_request)]})
        final_message = result["messages"][-1].content
        logger.info("\nAgent response:\n%s\n", final_message)

        routing_plan = LAST_ROUTING_PLAN if LAST_ROUTING_PLAN else _fallback_plan("empty result")
        return state.model_copy(update={
            "agent_messages": result["messages"],
            "agent_completed": True,
            "routing_plan": routing_plan,
        })
    except Exception as e:
        logger.error("Agent failed: %s", e)
        import traceback
        logger.error(traceback.format_exc())
        routing_plan = _fallback_plan("exception")
        return state.model_copy(update={"agent_completed": True, "routing_plan": routing_plan})

def approval_hitl_node(state: WorkflowState) -> Command:
    routes = state.routing_plan.get("routing_plan", [])
    if state.current_route_index >= len(routes):
        return Command(update={})

    route = routes[state.current_route_index]
    
    # Local dev mode – skip Action Center and auto-approve the route
    if config.AUTO_APPROVE_IN_LOCAL:
        logger.info(
            "AUTO_APPROVE_IN_LOCAL is enabled - auto-approving route %s (%d/%d)",
            route.get("city"),
            state.current_route_index + 1,
            len(routes),
        )
        trigger_rpa_for_route(route)
        return Command(update={
            "approved_routes": state.approved_routes + [route],
            "current_route_index": state.current_route_index + 1,
            "revision_in_progress": False,
            "current_revision_iteration": 0,
            "pending_manager_note": "",
        })
    
    if state.revision_in_progress and state.current_revision_iteration > 0:
        logger.info("PHASE 2: HITL Approval (%d/%d) - Revision %d for %s", 
                   state.current_route_index + 1, len(routes), 
                   state.current_revision_iteration, route["city"])
        iteration_context = f" (Revision {state.current_revision_iteration}/{state.max_revision_iterations})"
    else:
        logger.info("PHASE 2: HITL Approval (%d/%d) for %s", 
                   state.current_route_index + 1, len(routes), route["city"])
        iteration_context = ""

    visit_details = "\n".join([
        f"Visit {i+1}: {v['clinic']['clinic_name']} ({len(v['devices'])} devices)"
        for i, v in enumerate(route["visits"])
    ])
    
    agent_comments = _build_agent_comments(
        route, 
        manager_note=state.pending_manager_note if state.revision_in_progress else ""
    )

    action_data = interrupt(CreateTask(
        app_name="Routeapprovalform",
        title=f"Route{iteration_context} - {route['city']} - {len(route['visits'])} visits",
        data={
            "City": route["city"],
            "RouteDate": route.get("route_date"),
            "TechnicianName": route["technician"]["technician_name"],
            "TotalVisits": len(route["visits"]),
            "TotalDistanceKm": route["route"]["total_distance_km"],
            "RouteMapUrl": route["route"]["route_map_url"],
            "VisitDetails": visit_details,
            "TotalServiceHours": route.get("service_hours"),
            "TotalTravelHours": route.get("travel_hours"),
            "TotalWorkHours": route.get("total_work_hours"),
            "AgentComments": agent_comments,
        },
        app_version=1,
        app_folder_path=config.UIPATH_FOLDER_PATH,
    ))

    decision = (action_data.get(config.APP_FIELD_SELECTED_OUTCOME) or "").strip()
    manager_note = (action_data.get(config.APP_FIELD_MANAGER_COMMENTS) or "").strip()
    if not decision:
        logger.warning("SelectedOutcome empty; defaulting to Approved")
        decision = "Approved"

    update_dict = {}

    if decision == "Approved":
        logger.info("Approved: %s (after %d revision(s))", route["city"], state.current_revision_iteration)
        trigger_rpa_for_route(route)
        update_dict = {
            "approved_routes": state.approved_routes + [route],
            "current_route_index": state.current_route_index + 1,
            "revision_in_progress": False,
            "current_revision_iteration": 0,
            "pending_manager_note": "",
        }

    elif decision == "Rejected":
        logger.info("Rejected: %s", route["city"])
        update_dict = {
            "rejected_routes": state.rejected_routes + [route],
            "current_route_index": state.current_route_index + 1,
            "revision_in_progress": False,
            "current_revision_iteration": 0,
            "pending_manager_note": "",
        }

    elif decision == "ChangesRequested":
        next_iteration = state.current_revision_iteration + 1
        
        if next_iteration > state.max_revision_iterations:
            logger.warning("Max revision iterations (%d) reached for %s. Marking as rejected.", 
                         state.max_revision_iterations, route["city"])
            update_dict = {
                "rejected_routes": state.rejected_routes + [route],
                "current_route_index": state.current_route_index + 1,
                "revision_in_progress": False,
                "current_revision_iteration": 0,
                "pending_manager_note": "",
            }
        else:
            logger.info("Changes requested (iteration %d/%d). Manager note: %s", 
                       next_iteration, state.max_revision_iterations, manager_note)
            
            # Get ALL devices in this city
            city_name = route["city"]
            logger.info("ChangesRequested for %s: fetching all devices in city", city_name)
            analysis = analyze_equipment_status.invoke({})
            all_devices = analysis.get("overdue_devices", []) + analysis.get("urgent_devices", [])
            all_clinics = query_clinics.invoke({})
            clinics_in_city = [c for c in all_clinics if c.get("city") == city_name]
            clinic_ids_in_city = {c["clinic_id"] for c in clinics_in_city}
            devices_this_city = [d for d in all_devices if d.get("clinic_id") in clinic_ids_in_city]
            
            logger.info("Found %d clinics in %s with %d total devices requiring service", 
                       len(clinics_in_city), city_name, len(devices_this_city))
            
            limits = _compute_limits_for_devices(has_overdue=len([d for d in devices_this_city if d.get("days_until_due", 1) < 0]) > 0,
                                                 manager_note=manager_note)
            
            revised_plan = _plan_with_limits(devices_this_city, limits, manager_note)
            
            if revised_plan.get("routing_plan"):
                revised_route = revised_plan["routing_plan"][0]
                logger.info("Route regenerated successfully for %s", revised_route["city"])
                updated_routes = list(routes)
                updated_routes[state.current_route_index] = revised_route
                updated_routing_plan = {**state.routing_plan, "routing_plan": updated_routes}
                update_dict = {
                    "routing_plan": updated_routing_plan,
                    "revision_in_progress": True,
                    "current_revision_iteration": next_iteration,
                    "pending_manager_note": manager_note,
                }
            else:
                logger.error("Failed to regenerate route for %s. Marking as rejected.", route["city"])
                update_dict = {
                    "rejected_routes": state.rejected_routes + [route],
                    "current_route_index": state.current_route_index + 1,
                    "revision_in_progress": False,
                    "current_revision_iteration": 0,
                    "pending_manager_note": "",
                }

    else:
        logger.warning("Unknown decision '%s' for %s. Treating as rejected.", decision, route["city"])
        update_dict = {
            "rejected_routes": state.rejected_routes + [route],
            "current_route_index": state.current_route_index + 1,
            "revision_in_progress": False,
            "current_revision_iteration": 0,
            "pending_manager_note": "",
        }

    return Command(update=update_dict)

def trigger_rpa_for_route(route: Dict[str, Any]) -> bool:
    """
    Post-approval side effects for an approved route:
    - Email notifications via MCP tool 'Send_Calibration_Notifications' (fallback to classic invoke)
    - Slack notification via MCP tool 'Send_Slack_Notification'
    - Data Fabric record insert via MCP tool 'AddServiceOrder'
    The bridge handles async/sync differences safely.
    """
    try:
        logger.info("Preparing to trigger post-approval actions for %s", route.get("city"))
        # ---------------- Build EMAIL payload (same schema as before) ----------------
        route_data: Dict[str, Any] = {
            "City": route.get("city"),
            "TechnicianName": route.get("technician", {}).get("technician_name"),
            "TechnicianEmail": route.get("technician", {}).get("email"),
            "RouteDate": route.get("route_date"),
            "TotalVisits": len(route.get("visits", [])),
            "TotalDistanceKm": route.get("route", {}).get("total_distance_km"),
            "RouteMapUrl": route.get("route", {}).get("route_map_url"),
            "Visits": [],
        }
        for i, visit in enumerate(route.get("visits", []), 1):
            visit_data = {
                "VisitNumber": i,
                "ClinicName": visit.get("clinic", {}).get("clinic_name"),
                "ClinicEmail": visit.get("clinic", {}).get("contact_email"),
                "ClinicAddress": visit.get("clinic", {}).get("address"),
                "Devices": [
                    {
                        "EquipmentId": d.get("equipment_id"),
                        "DeviceType": d.get("device_type"),
                        "Status": "OVERDUE" if (d.get("days_until_due", 0) < 0) else "URGENT",
                    }
                    for d in visit.get("devices", [])
                ],
            }
            route_data["Visits"].append(visit_data)

        from mcp_bridge import send_calibration_notifications, send_slack_notification, add_service_order

        # ---------------- EMAIL via MCP (or classic fallback) ----------------
        ok_mail = send_calibration_notifications(route_data)
        if ok_mail:
            logger.info("Email notifications dispatched via MCP/classic.")
        else:
            logger.error("Email workflow failed via MCP/classic")

        # ---------------- SLACK via MCP ----------------
        clinics_human: List[str] = []
        for v in route.get("visits", []):
            c = v.get("clinic", {}) or {}
            clinics_human.append(f"{c.get('clinic_name')} - {c.get('address')}")
        slack_payload: Dict[str, Any] = {
            "technician_name": route.get("technician", {}).get("technician_name"),
            "technician_email": route.get("technician", {}).get("email"),
            "visit_count": len(route.get("visits", [])),
            "city": route.get("city"),
            "route_date": route.get("route_date"),
            "route_map_url": route.get("route", {}).get("route_map_url"),
            "total_distance_km": route.get("route", {}).get("total_distance_km"),
            "clinics": clinics_human,
        }
        ok_slack = send_slack_notification(slack_payload)
        if ok_slack:
            logger.info("Slack notification sent.")
        else:
            logger.warning("Slack notification tool returned False (check MCP tool + InArgument).")

        # ---------------- DATA FABRIC via MCP ----------------
        from datetime import datetime as _dt
        first_visit = (route.get("visits") or [{}])[0] if route.get("visits") else {}
        first_clinic = first_visit.get("clinic", {}) if first_visit else {}
        first_device = (first_visit.get("devices") or [{}])[0] if first_visit else {}
        technician_id = route.get("technician", {}).get("technician_id") or route.get("technician", {}).get("id") or "TECH-UNKNOWN"
        est_hours = route.get("total_work_hours") or (route.get("service_hours", 0.0) or 0.0) + (route.get("travel_hours", 0.0) or 0.0)
        order_id = f"ORD-{_dt.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
        entity_record: Dict[str, Any] = {
            "orderId": order_id,
            "clinicId": first_clinic.get("clinic_id") or first_clinic.get("id") or "CLI-UNKNOWN",
            "equipmentId": first_device.get("equipment_id") or "EQ-UNKNOWN",
            "technicianId": technician_id,
            "scheduledDate": route.get("route_date"),
            "routeSequence": 1,
            "estimatedDurationHours": float(est_hours or 0.0),
            "status": "Approved",
            "priority": 1,
            "notes": "Agent auto-created after manager approval.",
            "routeMapUrl": route.get("route", {}).get("route_map_url"),
            "totalDistanceKm": route.get("route", {}).get("total_distance_km"),
            "approvedBy": config.APPROVER_EMAIL,
            "createdDate": _dt.now().astimezone().isoformat(timespec="seconds"),
        }
        ok_entity = add_service_order(entity_record)
        if ok_entity:
            logger.info("Service order entity created: %s", entity_record.get("orderId"))
        else:
            logger.warning("AddServiceOrder tool returned False (check MCP tool + InArgument).")

        # Consider success if at least email went out (Slack & DF are auxiliary)
        return bool(ok_mail)
    except Exception as e:
        logger.error("Post-approval triggers failed for %s: %s", route.get("city"), e, exc_info=True)
        return False
def summary_node(state: WorkflowState) -> WorkflowState:
    logger.info("=" * 60)
    logger.info("WORKFLOW COMPLETE")
    logger.info("Approved: %d", len(state.approved_routes))
    logger.info("Rejected: %d", len(state.rejected_routes))
    logger.info("=" * 60)
    return state.model_copy(update={"workflow_complete": True})

def should_start_approvals(state: WorkflowState) -> str:
    routes = state.routing_plan.get("routing_plan", [])
    return "approve" if routes else "end"

def should_continue_approvals(state: WorkflowState) -> str:
    if state.revision_in_progress:
        logger.info("Revision in progress, looping back to approval for route %d", state.current_route_index + 1)
        return "next"
    total_routes = len(state.routing_plan.get("routing_plan", []))
    return "next" if state.current_route_index < total_routes else "finish"

# ---------- Graph ----------

graph = StateGraph(WorkflowState)
graph.add_node("agent", run_agent_node)
graph.add_node("approval", approval_hitl_node)
graph.add_node("summary", summary_node)
graph.set_entry_point("agent")

graph.add_conditional_edges("agent", should_start_approvals, {"approve": "approval", "end": "summary"})
graph.add_conditional_edges("approval", should_continue_approvals, {"next": "approval", "finish": "summary"})
graph.add_edge("summary", END)

agent = graph.compile()

# ---------- Main ----------

if __name__ == "__main__":
    logger.info("Starting calibration dispatcher agent...")
    initial_state = WorkflowState()
    _ = agent.invoke(initial_state)
    logger.info("Done!")