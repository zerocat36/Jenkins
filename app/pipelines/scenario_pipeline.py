from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.rag.vector_store import get_robot_context
from app.services.image_parser import parse_blueprint
from app.services.bottleneck_evaluator import analyze_risk
from app.services.visualizer import create_heatmap
from app.services.report_service import format_isaac_params, build_report


@dataclass(frozen=True)
class ScenarioPipelineResult:
    layout: dict
    robot_context: str
    risk_analysis: dict
    heatmap_path: Path | None
    report: dict
    isaac_api_parameters: dict


def run_scenario_pipeline(
    *,
    settings: Settings,
    blueprint_image_path: Path,
    scenario_text: str,
    top_k: int = 3,
) -> ScenarioPipelineResult:
    # 1) 이미지 파싱 + 시나리오 -> 레이아웃 JSON
    layout_json = parse_blueprint(
        image_path=str(blueprint_image_path),
        scenario_text=scenario_text,
        settings=settings,
    )

    # 2) RAG: 시나리오 기반으로 로봇 스펙 컨텍스트 검색
    robot_context = get_robot_context(
        query=scenario_text,
        settings=settings,
        k=top_k,
    )

    # 3) LLM 분석: 충돌/병목/위험지점 산출
    risk_analysis_result = analyze_risk(
        layout_json=layout_json,
        scenario_text=scenario_text,
        robot_context=robot_context,
        settings=settings,
    )

    # 4) 시각화: 설계도 위 히트맵 생성
    heatmap_path = None
    try:
        heatmap_path = Path(
            create_heatmap(
                image_path=str(blueprint_image_path),
                risk_analysis_result=risk_analysis_result,
                settings=settings,
            )
        )
    except Exception:
        # 시각화는 부가 산출물이라 실패해도 파이프라인은 계속 진행
        heatmap_path = None

    # 5) 보고서 + Isaac API 파라미터 포맷팅
    report = build_report(
        layout_json=layout_json,
        scenario_text=scenario_text,
        robot_context=robot_context,
        risk_analysis_result=risk_analysis_result,
    )
    image_size_px = {"width": 810, "height": 570}
    try:
        from PIL import Image
        with Image.open(blueprint_image_path) as img:
            image_size_px = {"width": img.width, "height": img.height}
    except Exception:
        pass
    isaac_params = format_isaac_params(
        layout_json=layout_json,
        risk_analysis_result=risk_analysis_result,
        image_path=blueprint_image_path.name,
        facility_name=layout_json.get("image_summary") or "물류센터 시뮬레이션",
        image_size_px=image_size_px,
        grid_size=15,
        scale_m_per_grid=2.0,
        total_episodes=500,
    )

    return ScenarioPipelineResult(
        layout=layout_json,
        robot_context=robot_context,
        risk_analysis=risk_analysis_result,
        heatmap_path=heatmap_path,
        report=report,
        isaac_api_parameters=isaac_params,
    )

