"""로봇 스펙 문서(PDF, TXT, DOCX) 추출 → 청킹 → 벡터 스토어 저장."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import Settings
from app.rag.vector_store import save_documents_to_store

ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".docx"})


def _extract_text_docx(path: Path) -> str:
    try:
        import docx2txt
        return (docx2txt.process(str(path)) or "").strip()
    except ImportError:
        raise ValueError(
            "DOCX 파일을 처리하려면 docx2txt 패키지가 필요합니다. "
            "pip install docx2txt 또는 PDF/TXT 파일을 사용해 주세요."
        )


def _extract_text_pdf_ocr(path: Path) -> str:
    """스캔 PDF(이미지)에서 OCR로 텍스트 추출. pdf2image + pytesseract 필요."""
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError as e:
        raise ValueError(
            "스캔 PDF를 처리하려면 pdf2image, pytesseract와 시스템에 poppler, tesseract-ocr이 필요합니다. "
            "pip install pdf2image pytesseract 후 poppler와 tesseract를 설치해 주세요."
        ) from e
    try:
        images = convert_from_path(str(path), dpi=200)
    except Exception as e:
        raise ValueError(f"PDF를 이미지로 변환할 수 없습니다 (poppler 필요): {e}") from e
    texts = []
    for img in images:
        try:
            t = pytesseract.image_to_string(img, lang="kor+eng")
        except Exception:
            t = pytesseract.image_to_string(img, lang="eng")
        texts.append((t or "").strip())
    return "\n\n".join(texts).strip()


def ingest_robot_specs(file_path: str, *, settings: Settings) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"지원하지 않는 파일 형식입니다. 사용 가능: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            text = "\n".join([(p.extract_text() or "") for p in reader.pages])
        except Exception as e:
            raise ValueError(f"PDF 읽기 실패: {e}") from e
        text = (text or "").strip()
        if len(text) < 50:
            # 스캔 PDF일 수 있음 → OCR 시도
            try:
                text_ocr = _extract_text_pdf_ocr(path)
                if len(text_ocr) >= 1:
                    text = text_ocr
                else:
                    raise ValueError(
                        "PDF에서 추출된 텍스트가 없거나 너무 적습니다. "
                        "스캔된 이미지 PDF는 OCR을 시도했으나 충분한 텍스트를 찾지 못했습니다. "
                        "더 선명한 스캔이거나 텍스트가 선택·복사 가능한 PDF를 사용해 주세요."
                    )
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    "PDF에서 추출된 텍스트가 없거나 너무 적습니다. "
                    "스캔 PDF인 경우 poppler와 tesseract-ocr을 설치한 뒤 다시 시도해 주세요. "
                    f"(OCR 오류: {e})"
                ) from e
    elif suffix == ".docx":
        text = _extract_text_docx(path)
        if len(text) < 1:
            raise ValueError("DOCX에서 추출된 텍스트가 없거나 너무 적습니다.")
    else:
        text = path.read_text(encoding="utf-8")
        text = (text or "").strip()

    if not text:
        raise ValueError("문서에서 추출된 텍스트가 없습니다.")

    docs = [Document(page_content=text, metadata={"source": path.name})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)

    result = save_documents_to_store(split_docs, settings=settings)
    result["vector_db_path"] = str(settings.vector_db_path)
    result["vector_db_faiss_dir"] = str(
        getattr(settings, "vector_db_faiss_dir", "")
    )
    return result

