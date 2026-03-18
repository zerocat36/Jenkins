"""충돌/병목 위험 지점 분석 — LLM 기반."""
from __future__ import annotations

from typing import List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.core.config import Settings


class RiskPoint(BaseModel):
    x: int = Field(description="지점의 X 좌표")
    y: int = Field(description="지점의 Y 좌표")
    risk_score: int = Field(description="위험도 점수 (0~100)")
    reason: str = Field(description="충돌 또는 병목 발생 이유")
    type: str = Field(description="유형 ('collision' 또는 'bottleneck')")

class RiskAnalysisResult(BaseModel):
    points: List[RiskPoint] = Field(description="분석된 위험 지점 목록")

def analyze_risk(*, layout_json: dict, scenario_text: str, robot_context: str, settings: Settings) -> dict:
    """LangChain 파이프라인을 통해 충돌 및 병목 지점을 분석합니다."""
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    parser = JsonOutputParser(pydantic_object=RiskAnalysisResult)
    
    template = """
    당신은 스마트 팩토리 로봇 동선 설계 전문가입니다.
    아래의 [공장 레이아웃 정보], [작업 시나리오], 그리고 RAG 시스템에서 추출한 [로봇 스펙]을 종합하여
    로봇이 주행할 때 예상되는 충돌 지점과 병목 현상 지점을 분석하세요.

    [공장 레이아웃 정보 (JSON)]: {layout}
    
    [작업 시나리오]: {scenario}
    
    [로봇 스펙 (RAG Context)]: {context}
    
    반드시 로봇의 회전 반경, 크기, 속도(스펙)를 고려하여 장애물과의 거리를 계산해 평가해야 합니다.
    
    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["layout", "scenario", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # LangChain LCEL (LangChain Expression Language) 체인 구성
    chain = prompt | llm | parser
    
    # 체인 실행
    result = chain.invoke({
        "layout": str(layout_json),
        "scenario": scenario_text,
        "context": robot_context
    })
    
    return result