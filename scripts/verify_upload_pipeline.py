#!/usr/bin/env python3
"""
문서 업로드 → 청킹 → 임베딩 → FAISS/벡터 스토어 흐름 검증 스크립트.

실행: cd AIOps-ver2 && python scripts/verify_upload_pipeline.py

필요: .env에 OPENAI_API_KEY 설정, faiss-cpu 또는 pickle 백엔드 사용 가능
"""
from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings
from app.rag.ingest import ingest_robot_specs
from app.rag.vector_store import get_robot_context


def main() -> int:
    settings = get_settings()
    if not settings.openai_api_key:
        print("❌ OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")
        return 1

    # 테스트용 TXT 파일 생성 (프로젝트 data/uploads)
    root = Path(__file__).resolve().parents[1]
    tmp = root / "data" / "uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    test_file = tmp / "verify_test_spec.txt"
    test_file.write_text(
        "로봇 스펙 문서\n\n"
        "AGV-01: 최대 속도 1.5m/s, 회전 반경 0.8m\n"
        "AGV-02: 최대 속도 1.2m/s, 회전 반경 1.0m\n"
        "작업 시나리오: A구역 → B구역 → 충전소 순회",
        encoding="utf-8",
    )

    print("1. ingest_robot_specs (텍스트 추출 → 청킹 → 임베딩 → 저장)")
    try:
        result = ingest_robot_specs(str(test_file), settings=settings)
        print(f"   ✓ chunks={result.get('chunks')}, backend={result.get('backend')}")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        return 1

    print("2. get_robot_context (작업 시나리오로 유사도 검색)")
    try:
        context = get_robot_context(
            query="AGV가 A구역에서 B구역으로 이동하는 시나리오",
            settings=settings,
            k=3,
        )
        if context:
            print(f"   ✓ 검색 결과 길이: {len(context)} chars")
        else:
            print("   ⚠ 검색 결과 없음 (벡터 스토어가 비어있을 수 있음)")
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        return 1

    test_file.unlink(missing_ok=True)
    print("\n✅ 파이프라인 검증 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
