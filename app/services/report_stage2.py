"""
2단계 보고서 생성: 1단계 Vision JSON → GPT-4o → 한국어 분석 보고서.
노트북 #4.factory_robot_analysis.ipynb report_prompt 기준.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import Settings


REPORT_SYSTEM_PROMPT = """공장 로봇 동선 분석 결과 JSON이 주어집니다.
아래 4가지 섹션으로 나누어, 실제 운영 보고서 수준의 한국어 리포트를 작성하세요.

0) 요약 (Executive Summary)
- 한 단락으로 이번 분석의 핵심 발견 사항을 요약합니다.
- 가장 위험한 구간 2~3개와 추천 조치를 간단히 언급합니다.

🔵 병목 구간 (Bottleneck)
- 각 항목: ID, 위치 설명, 발생 원인, 예상 영향

🔴 충돌 위험 구간 (Collision Risk)
- 각 항목: ID, 위치 설명, 충돌 시나리오, 위험도

🔀 동선 최적화 제안
- 위 분석을 바탕으로 경로 개선안 2~3가지 제시
- 각 제안은 구체적인 조치 방법 포함

4) 운영 인사이트 및 후속 액션
- 운영자가 다음으로 취해야 할 액션 아이템을 Bullet 형식으로 정리합니다.
- 예: 설계 수정, 로봇 스펙 보완, 교육/안전조치, Isaac Sim 시뮬레이션 대상 구간 등.

형식 가이드:
- 섹션 제목은 명확히 구분되도록 '## [섹션명]' 형식으로 적어주세요.
- 전체 분량은 최소 800자 이상이 되도록, 충분히 구체적으로 서술하세요.
- 불필요한 수사는 줄이고, 실제로 설계/운영자가 참고할 수 있는 실무적인 표현을 사용하세요."""


def run_report_from_vision_json(
    *,
    analysis_json: dict,
    rag_context: str = "",
    settings: Settings,
) -> str:
    """
    1단계 Vision 분석 JSON을 받아 한국어 분석 보고서를 생성합니다.
    rag_context: 작업 시나리오로 FAISS에서 검색한 로봇 스펙(해당 시나리오에 사용되는 로봇 관련 PDF 청크)을
                 병목/충돌 분석 보고서 작성에 반영합니다.
    """
    import json
    analysis_str = json.dumps(analysis_json, ensure_ascii=False, indent=2)
    if rag_context:
        analysis_str = f"[로봇 스펙 참고 자료]\n{rag_context[:3000]}\n\n[분석 JSON]\n{analysis_str}"

    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.25,
        max_tokens=2800,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", REPORT_SYSTEM_PROMPT),
        ("user", "분석 JSON:\n{analysis_json}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"analysis_json": analysis_str})
