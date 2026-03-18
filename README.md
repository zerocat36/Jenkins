# AIOps — Factory Robot Analysis API

rob-java 프론트엔드와 연동되는 공장 설계도·로봇 동선 분석 백엔드입니다.

## 워크플로우

1. **입력**: 공장 설계도 이미지, 작업 시나리오, 로봇 스펙, 로봇 문서(PDF/TXT)
2. **문서 임베딩**: 로봇 문서 → FAISS 기반 지식베이스 구축 (또는 pickle 폴백)
3. **1단계 분석**: GPT-4o Vision → 설계도에서 충돌·병목 후보를 **정규화 좌표 JSON**으로 출력
4. **2단계 분석**: 해당 JSON + RAG 컨텍스트 → 한국어 분석 보고서 생성 + 설계도 위에 **충돌(빨강)/병목(파랑)** 마커 오버레이 이미지

## API 엔드포인트 (prefix: `/v1`)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/health` | 헬스체크 |
| POST | `/upload-layout` | 설계도 이미지 업로드 |
| POST | `/upload-robot-specs` | 로봇 스펙 문서 업로드 및 FAISS/벡터 DB 저장 |
| POST | `/analyze/stage1` | 1단계 Vision 분석 (이미지 + 시나리오/스펙 → JSON) |
| POST | `/analyze/stage2` | 2단계 보고서·주석 이미지 (이미지 + stage1 JSON → report + base64 이미지) |
| POST | `/analyze/full` | 전체 파이프라인 한 번에 (stage1 + stage2) |
| POST | `/analyze-scenario` | (레거시) 기존 시나리오 파이프라인 |

### 1단계 요청/응답 예시

- **요청**: `POST /v1/analyze/stage1` — `blueprint_image` (파일), `scenario_text`, `robot_specs_text` (Form)
- **응답**: `layout_json` (image_summary, collision_candidates, bottleneck_candidates, 정규화 좌표 x,y ∈ [0,1])

### 2단계 요청/응답 예시

- **요청**: `POST /v1/analyze/stage2` — `blueprint_image` (파일), `stage1_json_str` (1단계에서 받은 JSON 문자열), `use_rag` (bool)
- **응답**: `report` (한국어 마크다운), `annotated_image_base64` (PNG base64)

## 환경 변수 (.env)

- `OPENAI_API_KEY`: OpenAI API 키 (필수)
- `CHAT_MODEL`, `VISION_MODEL`: 기본값 `gpt-4o` (Vision/보고서용)
- `EMBED_MODEL`: 임베딩 모델 (기본 `text-embedding-3-small`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: 문서 청크 설정
- `EMBED_BATCH_SIZE`: 임베딩 API 호출 시 배치 크기 (기본 100, 대량 청크 시 rate limit·타임아웃 완화)

스캔 PDF(이미지 PDF)를 쓰려면 OCR용 패키지와 시스템 도구가 필요합니다: `pip install pdf2image pytesseract`, macOS는 `brew install poppler tesseract tesseract-lang`.

**로컬에서 처음 셋업할 때**

```bash
# 프로젝트 루트에 .env 파일이 있어야 합니다. OPENAI_API_KEY 등을 설정하세요.
# AIOps 폴더에 이미 .env 가 있다면 복사해 사용할 수 있습니다.
cp ../AIOps/.env .env
```

## 실행

**로컬 (uvicorn)** — factory-robot-app은 8004로 연결됩니다.

```bash
cd AIOps-ver2
# 가상환경 사용 시 반드시 활성화 후 실행 (같은 환경에 pdf2image, pytesseract 있어야 스캔 PDF 동작)
# source venv/bin/activate  또는  conda activate pjt
pip install -r requirements.txt
# .env 설정 후 — 가상환경의 python으로 실행해야 OCR 패키지 인식됨
python -m uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

**Docker** (이미지 빌드 시 포트는 docker-compose.yaml 기준, 필요 시 8004:8003 등으로 변경)

```bash
cd AIOps-ver2
# .env 파일이 있어야 함 (OPENAI_API_KEY 등 설정)
docker compose up -d --build
docker compose logs -f paper-api
# 종료: docker compose down
```

프론트엔드(rob-java)의 `app.aiops.base-url`을 `http://localhost:8004`로 두면 위 API를 호출할 수 있습니다.  
**factory-robot-app**은 개발 시 Vite 프록시 `/v1` → `http://localhost:8004`로 이 API를 사용합니다.

## 참고

- 프롬프트·시각화 로직: `#4.factory_robot_analysis.ipynb` 기준
- FAISS 미설치 시 자동으로 pickle 기반 벡터 스토어 사용
