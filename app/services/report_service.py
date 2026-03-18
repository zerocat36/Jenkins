from __future__ import annotations

from typing import Any


def build_report(
    *,
    layout_json: dict,
    scenario_text: str,
    robot_context: str,
    risk_analysis_result: dict,
) -> dict:
    points = risk_analysis_result.get("points", []) if isinstance(risk_analysis_result, dict) else []

    collision = [p for p in points if p.get("type") == "collision"]
    bottleneck = [p for p in points if p.get("type") == "bottleneck"]

    def _top(points_list: list[dict[str, Any]], n: int = 5) -> list[dict[str, Any]]:
        return sorted(points_list, key=lambda x: int(x.get("risk_score", 0)), reverse=True)[:n]

    return {
        "scenario": scenario_text,
        "layout_summary": layout_json.get("layout_summary"),
        "robot_context_used": robot_context[:4000] if robot_context else "",
        "risk_table": [
            {
                "type": p.get("type"),
                "x": p.get("x"),
                "y": p.get("y"),
                "risk_score": p.get("risk_score"),
                "reason": p.get("reason"),
            }
            for p in _top(points, n=50)
        ],
        "highlights": {
            "top_collision_points": _top(collision, n=5),
            "top_bottleneck_points": _top(bottleneck, n=5),
        },
    }


def format_isaac_params(
    *,
    layout_json: dict,
    risk_analysis_result: dict,
    image_path: str = "",
    facility_name: str = "",
    image_size_px: dict | None = None,
    grid_size: int = 15,
    scale_m_per_grid: float = 2.0,
    session_id: str | None = None,
    total_episodes: int = 500,
) -> dict:
    """NVIDIA Isaac Sim 전달용 session_id + params 스키마로 구성합니다."""
    from app.services.isaac_schema import build_isaac_payload

    return build_isaac_payload(
        layout_json=layout_json,
        risk_analysis_result=risk_analysis_result,
        image_path=image_path,
        facility_name=facility_name,
        image_size_px=image_size_px,
        grid_size=grid_size,
        scale_m_per_grid=scale_m_per_grid,
        session_id=session_id,
        parse_source="manual_vision",
        total_episodes=total_episodes,
    )

