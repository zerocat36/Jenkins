"""앱 설정 — 환경변수 및 경로."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    # .../AIOps/app/core/config.py -> parents[2] == .../AIOps
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    uploads_dir: Path
    outputs_dir: Path
    vector_db_path: Path
    vector_db_faiss_dir: Path

    openai_api_key: str | None
    vision_model: str
    embed_model: str
    chat_model: str

    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int  # 임베딩 API 호출 시 배치 크기 (rate limit·타임아웃 방지)

    # Optional vLLM OpenAI-compatible endpoint for VLM.
    vllm_api_base: str | None
    vllm_model_name: str | None


def get_settings() -> Settings:
    load_dotenv(override=False)

    root = _project_root()
    data_dir = root / "data"
    uploads_dir = data_dir / "uploads"
    outputs_dir = data_dir / "outputs"
    vector_db_path = data_dir / "vector_db" / "robot_specs.pkl"
    vector_db_faiss_dir = data_dir / "vector_db" / "faiss"

    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "vector_db").mkdir(parents=True, exist_ok=True)
    vector_db_faiss_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        project_root=root,
        data_dir=data_dir,
        uploads_dir=uploads_dir,
        outputs_dir=outputs_dir,
        vector_db_path=vector_db_path,
        vector_db_faiss_dir=vector_db_faiss_dir,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
        vision_model=os.getenv("VISION_MODEL", "gpt-4o"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        embed_batch_size=int(os.getenv("EMBED_BATCH_SIZE", "100")),
        vllm_api_base=os.getenv("VLLM_API_BASE"),
        vllm_model_name=os.getenv("VLLM_MODEL_NAME"),
    )

