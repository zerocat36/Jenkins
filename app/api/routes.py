"""
Factory Robot Analysis API — 설계도/문서 업로드, FAISS 임베딩, Vision 분석, 보고서 생성.

rob-java, factory-robot-app 프론트엔드와 연동됩니다.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.deps import verify_session
from app.rag.ingest import ALLOWED_EXTENSIONS, ingest_robot_specs
from app.rag.vector_store import get_robot_context
from app.services.report_stage2 import run_report_from_vision_json
from app.services.vision_stage1 import (
    encode_image_from_path,
    parse_vision_json,
    run_vision_analysis,
)
from app.services.visualizer import draw_points_on_image_normalized

router = APIRouter()
settings = get_settings()
logger = logging.getLogger("aiops")


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------


def _save_uploaded_file(upload: UploadFile, base_name: str, dest_dir: Path) -> tuple[str, Path]:
    """업로드 파일을 저장하고 (safe_name, file_path) 반환."""
    safe_name = f"{uuid.uuid4().hex}_{os.path.basename(base_name)}"
    file_path = dest_dir / safe_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return safe_name, file_path


def _infer_spec_extension(filename: str | None, content_type: str | None) -> str:
    """파일명에 확장자가 없을 때 content_type으로 추정."""
    if filename and Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
        return Path(filename).suffix.lower()
    if content_type:
        if "pdf" in content_type:
            return ".pdf"
        if "wordprocessingml" in content_type or "msword" in content_type:
            return ".docx"
        if "text/plain" in content_type:
            return ".txt"
    return ""


def _resolve_spec_extension(filename: str, content_type: str) -> str:
    """지원 확장자 결정. 없으면 빈 문자열."""
    ext = Path(filename).suffix.lower() if filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        ext = _infer_spec_extension(filename or None, content_type or None)
    return ext if ext in ALLOWED_EXTENSIONS else ""


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------


@router.get("/health", summary="헬스체크")
async def health():
    return {"status": "ok"}


@router.post("/upload-layout", summary="설계도 이미지 업로드")
async def upload_layout(
    blueprint_image: UploadFile = File(...),
    _: str = Depends(verify_session),
):
    """공장 설계도 이미지(PNG/JPG)를 업로드하고 저장합니다."""
    if not blueprint_image.content_type or not blueprint_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    try:
        base = blueprint_image.filename or "layout.png"
        safe_name, file_path = _save_uploaded_file(
            blueprint_image, base, settings.uploads_dir
        )
        return {
            "status": "success",
            "filename": blueprint_image.filename,
            "saved_as": safe_name,
            "path": str(file_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-robot-specs", summary="로봇 스펙 문서 업로드 및 벡터 DB 저장")
async def upload_robot_specs(
    file: UploadFile = File(..., description="FormData 필드명 'file'로 전송. PDF, TXT, DOCX 지원"),
):
    """
    로봇 정보가 담긴 문서(PDF, TXT, DOCX)를 업로드하여 FAISS/벡터 DB에 임베딩합니다.
    쿠키/세션 없이 호출 가능합니다.
    """
    filename = file.filename or ""
    content_type = file.content_type or ""
    logger.info("upload-robot-specs filename=%s content_type=%s", filename, content_type)

    ext = _resolve_spec_extension(filename, content_type)
    if not ext:
        detail = (
            f"지원 형식이 아닙니다. PDF, TXT, DOCX만 가능합니다. "
            f"(파일명: {filename or '비어 있음'}, content-type: {content_type or '없음'})"
        )
        logger.warning("upload-robot-specs 400: %s", detail)
        raise HTTPException(status_code=400, detail=detail)

    base = os.path.basename(filename or "doc")
    if not base.lower().endswith(ext):
        base = (base or "doc") + ext

    try:
        safe_name, file_path = _save_uploaded_file(file, base, settings.uploads_dir)
        result = await asyncio.to_thread(
            ingest_robot_specs, str(file_path), settings=settings
        )
        return {
            "status": "success",
            "filename": file.filename,
            "saved_as": safe_name,
            "ingest": result,
        }
    except FileNotFoundError as e:
        logger.warning("upload-robot-specs file not found: %s", e)
        raise HTTPException(status_code=500, detail="업로드 디렉터리를 찾을 수 없습니다.")
    except ValueError as e:
        logger.warning("upload-robot-specs 400: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        err_msg = str(e)
        if "extract_text" in err_msg or "PdfReader" in err_msg or "pypdf" in err_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="PDF에서 텍스트를 추출할 수 없습니다. 텍스트가 선택 가능한 PDF를 사용해 주세요.",
            )
        if "embed" in err_msg.lower() or "openai" in err_msg.lower() or "api_key" in err_msg.lower():
            raise HTTPException(
                status_code=500,
                detail="임베딩 처리 중 오류가 발생했습니다. OpenAI API 설정을 확인해 주세요.",
            )
        logger.exception("upload_robot_specs 실패")
        raise HTTPException(status_code=500, detail=f"문서 처리 실패: {err_msg}")


@router.post("/analyze/stage1", summary="1단계 Vision 분석 (설계도 → 충돌/병목 JSON)")
async def analyze_stage1(
    blueprint_image: UploadFile = File(...),
    scenario_text: str = Form(""),
    robot_specs_text: str = Form(""),
    _: str = Depends(verify_session),
):
    """
    공장 설계도 이미지를 업로드하고, GPT-4o Vision으로 충돌·병목 후보를 정규화 좌표 JSON으로 반환합니다.
    시나리오/로봇 스펙 텍스트를 함께 주면 분석에 반영됩니다.
    """
    try:
        safe_name, image_path = _save_uploaded_file(
            blueprint_image, blueprint_image.filename or "layout.png", settings.uploads_dir
        )
        image_b64 = encode_image_from_path(image_path)
        layout_json = run_vision_analysis(
            image_b64=image_b64,
            scenario_text=scenario_text or "",
            robot_specs_text=robot_specs_text or "",
            settings=settings,
        )
        return {
            "status": "success",
            "layout_json": layout_json,
            "blueprint_saved_as": safe_name,
        }
    except Exception as e:
        logger.exception("analyze_stage1 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"1단계 분석 오류: {str(e)}")


@router.post("/analyze/stage2", summary="2단계 보고서·주석 이미지 생성")
async def analyze_stage2(
    blueprint_image: UploadFile = File(...),
    stage1_json_str: str = Form(...),
    scenario_text: str = Form(""),
    use_rag: bool = Form(True),
    _: str = Depends(verify_session),
):
    """
    1단계에서 받은 layout_json과 동일한 설계도 이미지를 전달하면,
    한국어 분석 보고서와 충돌/병목이 표시된 이미지(base64)를 반환합니다.
    use_rag=True일 때, scenario_text를 쿼리로 FAISS에서 작업 시나리오에 해당하는 로봇 스펙만 RAG로 가져와 보고서 LLM에 전달합니다.
    """
    try:
        safe_name, image_path = _save_uploaded_file(
            blueprint_image, blueprint_image.filename or "layout.png", settings.uploads_dir
        )
        analysis = parse_vision_json(stage1_json_str)
        rag_context = ""
        if use_rag:
            # 입력된 작업 시나리오 기준으로 해당 로봇 스펙만 검색 (시나리오에 사용되는 로봇 관련 내용만)
            query = (scenario_text or "").strip() or analysis.get("image_summary", "") or "로봇 스펙"
            rag_context = get_robot_context(query=query, settings=settings, k=5)
        report = run_report_from_vision_json(
            analysis_json=analysis,
            rag_context=rag_context,
            settings=settings,
        )
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"annotated_{safe_name}"
        if not out_filename.lower().endswith(".png"):
            out_filename += ".png"
        out_path = settings.outputs_dir / out_filename
        draw_points_on_image_normalized(
            image_path,
            analysis,
            out_path=out_path,
        )
        with open(out_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "status": "success",
            "report": report,
            "annotated_image_base64": image_base64,
            "annotated_image_path": str(out_path),
            "rag_used": bool(rag_context),
        }
    except Exception as e:
        logger.exception("analyze_stage2 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"2단계 분석 오류: {str(e)}")


@router.post("/analyze/full", summary="전체 파이프라인 (1단계 + 2단계) 한 번에 실행")
async def analyze_full(
    blueprint_image: UploadFile = File(...),
    scenario_text: str = Form(""),
    robot_specs_text: str = Form(""),
    use_rag: bool = Form(True),
    _: str = Depends(verify_session),
):
    """
    설계도 이미지와 시나리오(선택)를 받아 1단계 Vision → 2단계 보고서·주석 이미지까지 한 번에 수행합니다.
    use_rag=True일 때, 입력된 작업 시나리오를 쿼리로 FAISS에서 해당 시나리오에 사용되는 로봇 스펙만 RAG로 가져와
    Vision(병목/충돌 분석)과 보고서 LLM에 전달합니다.
    반환: stage1_json, report, annotated_image_base64, isaac_api_parameters.
    """
    try:
        safe_name, image_path = _save_uploaded_file(
            blueprint_image, blueprint_image.filename or "layout.png", settings.uploads_dir
        )
        image_b64 = encode_image_from_path(image_path)

        # RAG: 작업 시나리오 기준으로 해당 로봇 스펙만 검색 (시나리오에 사용되는 로봇에 해당하는 PDF 내용만)
        rag_context = ""
        if use_rag:
            query = (scenario_text or "").strip() or "로봇 설계"
            rag_context = get_robot_context(query=query, settings=settings, k=5)

        # 1단계 Vision: 시나리오 + RAG로 가져온 로봇 스펙을 함께 전달해 병목/충돌 분석
        layout_json = run_vision_analysis(
            image_b64=image_b64,
            scenario_text=scenario_text or "",
            robot_specs_text=rag_context or robot_specs_text or "",
            settings=settings,
        )
        report = run_report_from_vision_json(
            analysis_json=layout_json,
            rag_context=rag_context,
            settings=settings,
        )
        settings.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_filename = f"annotated_{safe_name}"
        if not out_filename.lower().endswith(".png"):
            out_filename += ".png"
        out_path = settings.outputs_dir / out_filename
        draw_points_on_image_normalized(image_path, layout_json, out_path=out_path)
        with open(out_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # 병목/충돌 분석 결과로 Isaac Sim 연계 페이로드 생성 (isaac_schema와 동일 스키마)
        isaac_api_parameters = None
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                image_size_px = {"width": img.width, "height": img.height}
        except Exception:
            image_size_px = {"width": 810, "height": 570}
        try:
            from app.services.isaac_schema import build_isaac_payload
            # vision 결과 = collision_candidates, bottleneck_candidates 포함 → risk로 사용
            # layout는 objects 없이 image_summary만 전달 (occupancy는 비움)
            layout_for_isaac = {"image_summary": layout_json.get("image_summary") or "", "objects": []}
            isaac_api_parameters = build_isaac_payload(
                layout_json=layout_for_isaac,
                risk_analysis_result=layout_json,
                image_path=safe_name,
                facility_name=layout_json.get("image_summary") or "물류센터 시뮬레이션",
                image_size_px=image_size_px,
                grid_size=15,
                scale_m_per_grid=2.0,
                total_episodes=500,
            )
        except Exception as e:
            logger.warning("Isaac payload 생성 건너뜀: %s", e)

        return JSONResponse(content={
            "status": "success",
            "stage1": {"layout_json": layout_json, "blueprint_saved_as": safe_name},
            "report": report,
            "annotated_image_base64": image_base64,
            "annotated_image_path": str(out_path),
            "rag_used": bool(rag_context),
            "isaac_api_parameters": isaac_api_parameters,
        })
    except Exception as e:
        logger.exception("analyze_full 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"전체 분석 오류: {str(e)}")


@router.post("/analyze-scenario", summary="(레거시) 설계도·시나리오 분석 파이프라인")
async def analyze_scenario(
    blueprint_image: UploadFile = File(...),
    scenario_text: str = Form(...),
    _: str = Depends(verify_session),
):
    """기존 시나리오 파이프라인(이미지 파싱 + RAG + 위험 분석 + 히트맵)을 실행합니다."""
    try:
        from app.pipelines.scenario_pipeline import run_scenario_pipeline

        safe_name, image_path = _save_uploaded_file(
            blueprint_image, blueprint_image.filename or "layout.png", settings.uploads_dir
        )
        pipeline_result = run_scenario_pipeline(
            settings=settings,
            blueprint_image_path=Path(image_path),
            scenario_text=scenario_text,
        )
        return JSONResponse(content={
            "status": "success",
            "inputs": {"scenario_text": scenario_text, "blueprint_image_saved_as": safe_name},
            "layout_json": pipeline_result.layout,
            "rag": {"robot_context": pipeline_result.robot_context},
            "risk_analysis": pipeline_result.risk_analysis,
            "visuals": {"heatmap_path": str(pipeline_result.heatmap_path) if pipeline_result.heatmap_path else None},
            "report": pipeline_result.report,
            "isaac_api_parameters": pipeline_result.isaac_api_parameters,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
