"""설계도 이미지 + 시나리오 → 레이아웃 JSON (vLLM 또는 OpenAI)."""
from __future__ import annotations

import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import Settings

def encode_image(image_path: str) -> str:
    """이미지를 Base64 문자열로 인코딩합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_blueprint(*, image_path: str, scenario_text: str, settings: Settings) -> dict:
    """설계도 이미지 + 작업 시나리오로 레이아웃 JSON을 추출합니다.

    - `settings.vllm_api_base`가 있으면 OpenAI 호환 vLLM 엔드포인트로 호출
    - 없으면 기본 OpenAI 호출
    """
    base64_image = encode_image(image_path)
    
    llm_kwargs = {
        "model": settings.vllm_model_name or settings.chat_model,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    if settings.vllm_api_base:
        llm_kwargs.update(
            {
                "base_url": settings.vllm_api_base,
                "api_key": "EMPTY",
            }
        )
    else:
        llm_kwargs.update({"api_key": settings.openai_api_key})

    llm = ChatOpenAI(**llm_kwargs)
    
    parser = JsonOutputParser()
    
    # 비전 모델을 위한 메시지 구성
    prompt_text = f"""
    당신은 공장 설계도(2D) 분석 AI입니다. 주어진 설계도 이미지와 작업 시나리오를 바탕으로
    로봇 스테이션, 장애물, 통로, 작업 구역 등 주요 객체들의 (x, y) 좌표를 가능한 한 많이 추출하세요.
    좌표는 "이미지 픽셀 좌표계" 기준으로 정수로 추정하여 반환하세요.

    작업 시나리오:
    {scenario_text}
    
    반드시 JSON만 출력합니다. 출력 형식:
    {parser.get_format_instructions()}
    """
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    )
    
    # 체인 실행 및 JSON 파싱
    response = llm.invoke([message])
    parsed_json = parser.parse(response.content)
    
    return parsed_json