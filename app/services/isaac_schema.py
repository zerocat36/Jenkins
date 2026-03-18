"""NVIDIA Isaac Sim 전달용 스키마 생성.

출력 형식:
  session_id, total_episodes, params { grid_size, wall_map, zones, agents, risk_zones, lr, gamma, ... }
"""
from __future__ import annotations

import re
import uuid
from typing import Any


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _norm_to_grid(x_norm: float, y_norm: float, grid_size: int) -> tuple[int, int]:
    gx = int(_clamp(x_norm, 0.0, 1.0) * grid_size)
    gy = int(_clamp(y_norm, 0.0, 1.0) * grid_size)
    gx = min(gx, grid_size - 1)
    gy = min(gy, grid_size - 1)
    return (gx, gy)


def _rasterize_occupancy(objects: list[dict], grid_size: int) -> list[list[int]]:
    """객체 bbox로부터 occupancy 그리드 생성."""
    grid = [[0] * grid_size for _ in range(grid_size)]
    for obj in objects:
        bbox = obj.get("bbox") or {}
        try:
            x1 = _clamp(float(bbox.get("x1", 0)), 0, 1)
            y1 = _clamp(float(bbox.get("y1", 0)), 0, 1)
            x2 = _clamp(float(bbox.get("x2", 0)), 0, 1)
            y2 = _clamp(float(bbox.get("y2", 0)), 0, 1)
        except (TypeError, ValueError):
            continue
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        i1 = int(x1 * grid_size)
        j1 = int(y1 * grid_size)
        i2 = min(int(x2 * grid_size) + 1, grid_size)
        j2 = min(int(y2 * grid_size) + 1, grid_size)
        for j in range(max(0, j1), min(grid_size, j2)):
            for i in range(max(0, i1), min(grid_size, i2)):
                grid[j][i] = 1
    return grid


def _ensure_border_walls(grid: list[list[int]], grid_size: int) -> list[list[int]]:
    """빈 그리드일 때 테두리 벽 추가."""
    has_wall = any(c == 1 for row in grid for c in row)
    if not has_wall and grid_size >= 3:
        for i in range(grid_size):
            grid[0][i] = 1
            grid[grid_size - 1][i] = 1
        for j in range(grid_size):
            grid[j][0] = 1
            grid[j][grid_size - 1] = 1
    return grid


def _layout_objects_to_zones_new(
    objects: list[dict],
    grid_size: int,
) -> list[dict]:
    """layout objects -> zones: {id, type, pos: [x,y], label}."""
    zones: list[dict] = []
    label_to_type = {
        "station": "workstation",
        "shelf": "workstation",
        "conveyor": "workstation",
        "path": "workstation",
        "intersection": "workstation",
        "obstacle": "workstation",
        "gate": "workstation",
        "charging": "charger",
        "other": "workstation",
    }
    seen_ids: set[str] = set()

    def _safe_id(base: str) -> str:
        rid = re.sub(r"[^\w가-힣]", "_", base)[:30] or "Z"
        if rid in seen_ids:
            for k in range(1, 100):
                cand = f"{rid}_{k}"
                if cand not in seen_ids:
                    seen_ids.add(cand)
                    return cand
        seen_ids.add(rid)
        return rid

    for idx, obj in enumerate(objects):
        bbox = obj.get("bbox") or {}
        try:
            x1 = float(bbox.get("x1", 0))
            y1 = float(bbox.get("y1", 0))
            x2 = float(bbox.get("x2", 0))
            y2 = float(bbox.get("y2", 0))
        except (TypeError, ValueError):
            continue
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx, gy = _norm_to_grid(cx, cy, grid_size)
        name = (obj.get("notes") or obj.get("id") or f"zone_{idx+1}")[:60]
        label_raw = str(obj.get("label", "other")).lower()
        zone_type = label_to_type.get(label_raw, "workstation")
        zones.append({
            "id": _safe_id(name).upper().replace(" ", "_"),
            "type": zone_type,
            "pos": [gx, gy],
            "label": name,
        })
    return zones


def _fallback_zones(grid_size: int) -> list[dict]:
    """객체 없을 때 기본 zones (충전소 + 작업대)."""
    s = grid_size - 1
    return [
        {"id": "CHARGER_A", "type": "charger", "pos": [1, 1], "label": "충전소 A (좌상단)"},
        {"id": "CHARGER_B", "type": "charger", "pos": [s - 1, 1], "label": "충전소 B (우상단)"},
        {"id": "CHARGER_C", "type": "charger", "pos": [1, s - 1], "label": "충전소 C (좌하단)"},
        {"id": "CHARGER_D", "type": "charger", "pos": [s - 1, s - 1], "label": "충전소 D (우하단)"},
        {"id": "WS_1", "type": "workstation", "pos": [s // 4, s // 4], "label": "작업대 1"},
        {"id": "WS_2", "type": "workstation", "pos": [3 * s // 4, s // 4], "label": "작업대 2"},
        {"id": "WS_3", "type": "workstation", "pos": [s // 4, 3 * s // 4], "label": "작업대 3"},
        {"id": "WS_4", "type": "workstation", "pos": [3 * s // 4, 3 * s // 4], "label": "작업대 4"},
        {"id": "WS_5", "type": "workstation", "pos": [s // 2, s // 3], "label": "작업대 5 (메인)"},
        {"id": "WS_6", "type": "workstation", "pos": [s // 2, 2 * s // 3], "label": "작업대 6"},
    ]


def _zone_by_id(zones: list[dict]) -> dict[str, dict]:
    return {z["id"]: z for z in zones}


def _risk_to_risk_zones(risk_analysis: dict, grid_size: int) -> list[dict]:
    """collision/bottleneck candidates -> risk_zones: {id, pos, radius, severity, label}."""
    out: list[dict] = []
    points: list[dict] = []

    for c in risk_analysis.get("collision_candidates", []):
        points.append({
            "x": float(c.get("x", 0)), "y": float(c.get("y", 0)),
            "reason": c.get("reason", ""), "id": c.get("id", ""),
            "severity": str(c.get("severity", "medium")).lower(),
            "risk_type": "collision",
        })
    for b in risk_analysis.get("bottleneck_candidates", []):
        points.append({
            "x": float(b.get("x", 0)), "y": float(b.get("y", 0)),
            "reason": b.get("reason", ""), "id": b.get("id", ""),
            "severity": str(b.get("severity", "medium")).lower(),
            "risk_type": "bottleneck",
        })
    for p in risk_analysis.get("points", []):
        x, y = float(p.get("x", 0)), float(p.get("y", 0))
        if x > 1 or y > 1:
            x, y = x / grid_size, y / grid_size
        points.append({
            "x": x, "y": y, "reason": p.get("reason", ""), "id": p.get("id", ""),
            "severity": "high" if p.get("type") == "collision" else "medium",
            "risk_type": p.get("type", "bottleneck"),
        })

    sev_to_radius = {"critical": 1.2, "high": 1.2, "danger": 0.8, "medium": 0.8, "warn": 1.0, "low": 1.0}
    sev_map = {"critical": "critical", "high": "critical", "danger": "danger", "medium": "danger", "warn": "warn", "low": "warn"}

    for i, p in enumerate(points):
        try:
            gx, gy = _norm_to_grid(float(p["x"]), float(p["y"]), grid_size)
            sev = str(p.get("severity", "medium")).lower()
            severity = sev_map.get(sev, "danger")
            radius = sev_to_radius.get(sev, 0.8)
            reason = (p.get("reason") or "")[:50] or "위험 구역"
            rid = p.get("id") or f"RZ_{i+1:02d}"
            rid = re.sub(r"[^\w]", "_", rid)[:30] or f"RZ_{i+1:02d}"
            out.append({
                "id": rid,
                "pos": [gx, gy],
                "radius": radius,
                "severity": severity,
                "label": reason,
            })
        except (TypeError, ValueError):
            continue
    return out


def _build_agents(zones: list[dict], grid_size: int) -> list[dict]:
    """zones로부터 agents: {id, label, spawn, goals} 생성."""
    zone_map = _zone_by_id(zones)
    chargers = [z for z in zones if z.get("type") == "charger"]
    workstations = [z for z in zones if z.get("type") == "workstation"]

    if not chargers:
        chargers = [zones[0]] if zones else [{"id": "CHARGER", "pos": [1, 1]}]
    if not workstations:
        workstations = zones[1:5] if len(zones) > 1 else []

    agents: list[dict] = []
    labels = [
        "입고→분류→충전 루트",
        "분류→포장→충전 루트",
        "포장→출고→충전 루트",
        "반품 처리 전담 루트",
    ]
    for i in range(min(4, max(1, len(chargers)))):
        ch = chargers[i % len(chargers)]
        spawn = ch.get("pos", [1, 1])
        goals: list[list[int]] = []
        for j in range(3):
            if workstations:
                ws = workstations[(i * 2 + j) % len(workstations)]
                goals.append(ws.get("pos", [2, 2]))
        goals.append(spawn)  # return to charger
        agents.append({
            "id": f"AGV_{i+1:02d}",
            "label": labels[i] if i < len(labels) else f"AGV {i+1} 루트",
            "spawn": spawn,
            "goals": goals,
        })
    return agents


def build_isaac_payload(
    *,
    layout_json: dict,
    risk_analysis_result: dict,
    image_path: str = "",
    facility_name: str = "",
    image_size_px: dict[str, int] | None = None,
    grid_size: int = 15,
    scale_m_per_grid: float = 2.0,
    session_id: str | None = None,
    parse_source: str = "manual_vision",
    timesteps: int = 200000,
    total_episodes: int = 500,
    lr: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    max_steps: int = 300,
) -> dict:
    """
    NVIDIA Isaac Sim 전달용 페이로드.

    반환: session_id, total_episodes, params { grid_size, wall_map, zones, agents, risk_zones, lr, gamma, ... }
    """
    risk_analysis_result = risk_analysis_result or {}
    layout_json = layout_json or {}
    objects = layout_json.get("objects", []) if isinstance(layout_json, dict) else []

    wall_map = _rasterize_occupancy(objects, grid_size)
    wall_map = _ensure_border_walls(wall_map, grid_size)

    if objects:
        zones = _layout_objects_to_zones_new(objects, grid_size)
        # 충전소가 없으면 코너에 추가
        chargers = [z for z in zones if z.get("type") == "charger"]
        if not chargers and len(zones) >= 4:
            s = grid_size - 1
            zones = [
                {"id": "CHARGER_A", "type": "charger", "pos": [1, 1], "label": "충전소 A"},
                {"id": "CHARGER_B", "type": "charger", "pos": [s - 1, 1], "label": "충전소 B"},
                {"id": "CHARGER_C", "type": "charger", "pos": [1, s - 1], "label": "충전소 C"},
                {"id": "CHARGER_D", "type": "charger", "pos": [s - 1, s - 1], "label": "충전소 D"},
            ] + zones
    else:
        zones = _fallback_zones(grid_size)

    agents = _build_agents(zones, grid_size)
    risk_zones = _risk_to_risk_zones(risk_analysis_result, grid_size)

    sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"

    return {
        "session_id": sid,
        "total_episodes": total_episodes,
        "params": {
            "grid_size": grid_size,
            "wall_map": wall_map,
            "zones": zones,
            "agents": agents,
            "risk_zones": risk_zones,
            "lr": lr,
            "gamma": gamma,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_min": epsilon_min,
            "max_steps": max_steps,
        },
    }
