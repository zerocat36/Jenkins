"""
1단계 Vision 분석: 공장 설계도 이미지 → GPT-4o Vision → 충돌/병목 후보 JSON (정규화 좌표).
노트북 #4.factory_robot_analysis.ipynb 프롬프트 기준.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import Settings


VISION_SYSTEM_PROMPT = """You are a senior robotics logistics engineer specializing in multi-robot warehouse and factory layout analysis.
You will receive ONE image showing robots, paths, shelves, conveyors, and intersections.

## DETECTION CRITERIA

### Collision Risk — look for ALL of the following:
- Intersections where 2+ robot paths cross without traffic control
- Narrow corridors (<2 robot widths) with bidirectional traffic
- Merge points where multiple lanes converge into one
- Blind corners or shelves blocking line-of-sight
- Head-on path conflicts on single-lane routes

### Bottleneck Risk — look for ALL of the following:
- Single-lane passages that are the only route between zones
- Chokepoints where 3+ robot routes pass through the same tile
- Loading/unloading stations with only one entry/exit
- Conveyor endpoints where robots queue to pick/place
- Door or gate openings that restrict simultaneous throughput

## OUTPUT FORMAT
Return ONLY valid JSON. No markdown, no explanation, no extra text.
Coordinate format: normalized x,y in [0,1], (0,0) = top-left, (1,1) = bottom-right.

Severity levels:
- "critical" : immediate risk, likely to cause stoppage or collision
- "high"     : frequent disruption expected under normal load
- "medium"   : occasional disruption under peak load
- "low"      : minor inefficiency, worth monitoring

Schema:
{
  "image_summary": "string (describe layout, zone count, robot count if visible, overall flow direction)",
  "collision_candidates": [
    {
      "id": "C1",
      "x": 0.0,
      "y": 0.0,
      "severity": "critical|high|medium|low",
      "priority": 1,
      "reason": "string (specific structural reason)",
      "suggested_fix": "string (one-line actionable suggestion)"
    }
  ],
  "bottleneck_candidates": [
    {
      "id": "B1",
      "x": 0.0,
      "y": 0.0,
      "severity": "critical|high|medium|low",
      "priority": 1,
      "reason": "string (specific structural reason)",
      "suggested_fix": "string (one-line actionable suggestion)"
    }
  ]
}

Rules:
- Return 2~6 items each for collision_candidates and bottleneck_candidates.
- priority: 1 = most urgent, ascending order within each list.
- If unsure, provide best-guess with reason "uncertain - [describe what you see]".
- JSON must be parseable by Python json.loads().
- Do NOT include any text outside the JSON object."""


def encode_image_from_path(image_path: str | Path) -> str:
    """파일 경로에서 이미지를 읽어 base64 문자열로 반환합니다."""
    path = Path(image_path)
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_image_from_bytes(image_bytes: bytes) -> str:
    """바이트 데이터를 base64 문자열로 반환합니다."""
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_vision_json(json_str: str) -> dict[str, Any]:
    """Vision 체인 출력 JSON 파싱. 마크다운/설명 텍스트가 섞여 있어도 최대한 복구."""
    cleaned = json_str.strip()

    # 코드블록 제거
    if cleaned.startswith("```"):
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0].strip()

    # 1차 시도: 그대로 파싱
    try:
        return json.loads(cleaned.strip())
    except Exception:
        pass

    # 2차 시도: 문자열 안에서 가장 바깥쪽 { ... } 블록만 추출
    try:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            return json.loads(snippet)
    except Exception:
        pass

    # 마지막 실패 시: 최소한의 스켈레톤 반환 (후속 단계에서 500을 피하기 위함)
    return {
        "image_summary": "Vision 모델 출력 파싱 실패. 원본 응답 일부: "
        + cleaned[:300],
        "collision_candidates": [],
        "bottleneck_candidates": [],
    }


def run_vision_analysis(
    *,
    image_b64: str,
    scenario_text: str = "",
    robot_specs_text: str = "",
    settings: Settings,
) -> dict[str, Any]:
    """
    설계도 이미지(base64)와 선택적 시나리오/로봇 스펙으로 1단계 Vision 분석을 수행합니다.
    반환: image_summary, collision_candidates, bottleneck_candidates (정규화 좌표).
    """
    llm = ChatOpenAI(
        model=getattr(settings, "vision_model", "gpt-4o") or settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
        max_tokens=2048,
    )
    # 시나리오/로봇 스펙은 텍스트로만 추가하고,
    # 항상 VISION_SYSTEM_PROMPT에서 정의한 JSON 스키마를 따르도록 고정.
    parts: list[str] = []
    if scenario_text:
        parts.append(f"작업 시나리오:\n{scenario_text}")
    if robot_specs_text:
        parts.append(f"로봇 스펙 참고:\n{robot_specs_text[:2000]}")
    parts.append(
        "위 정보를 참고하되, 시스템 지침의 스키마에 맞춰 "
        "collision_candidates 와 bottleneck_candidates 를 반드시 포함한 JSON만 출력하세요."
    )
    combined_text = "\n\n".join(parts)

    messages = [
        SystemMessage(content=VISION_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": combined_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ]
        ),
    ]
    response = llm.invoke(messages)
    return parse_vision_json(response.content)
