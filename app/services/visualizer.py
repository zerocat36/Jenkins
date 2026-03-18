"""
설계도 위에 충돌/병목 지점을 시각화합니다.
- create_heatmap: 기존 픽셀 좌표 + risk_score 기반 히트맵
- draw_points_on_image_normalized: 1단계 Vision JSON(정규화 좌표) 기반 오버레이 (노트북 #4 기준)
"""
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Tuple

from app.core.config import Settings


def _rgba(color: Tuple[int, int, int], alpha: int) -> Tuple[int, int, int, int]:
    r, g, b = color
    return (int(r), int(g), int(b), int(alpha))


def create_heatmap(*, image_path: str, risk_analysis_result: dict, settings: Settings) -> str:
    """
    원본 설계도 이미지 위에 LLM이 분석한 위험 지점을 히트맵 형태로 시각화합니다.
    
    Args:
        image_path (str): 원본 이미지 파일 경로
        risk_analysis_result (dict): LLM이 분석한 위험 지점 데이터 (x, y, risk_score, type 등)
        
    Returns:
        str: 시각화가 완료된 결과물 이미지의 저장 경로
    """
    # Pillow/FAISS가 macOS 환경에서 네이티브 크래시를 일으키는 경우가 있어,
    # OpenCV 대신 Pillow 기반으로 히트맵을 생성합니다.
    from PIL import Image, ImageDraw, ImageFont

    base = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    label_draw = ImageDraw.Draw(base, "RGBA")
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 2. 위험 지점 순회 및 그리기
    points = risk_analysis_result.get("points", [])
    
    for point in points:
        x, y = int(point.get("x", 0)), int(point.get("y", 0))
        risk_score = int(point.get("risk_score", 0))
        point_type = point.get("type", "unknown")
        
        # 위험도 점수에 따른 반경 설정 (위험할수록 범위가 넓어짐)
        radius = int(20 + (risk_score * 0.5))
        
        # 유형 및 위험도에 따른 색상 설정 (RGB)
        if point_type == "collision":
            color = (255, 60, 60)
        elif point_type == "bottleneck":
            color = (255, 170, 0)
        else:
            color = (255, 255, 0)
            
        # 히트맵 원(반투명)
        a = max(40, min(180, int(40 + risk_score * 1.4)))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=_rgba(color, a),
            outline=_rgba(color, min(255, a + 40)),
            width=2,
        )
        
        # 텍스트 라벨링 (점수 및 이유)
        label = f"{risk_score} - {point_type}"
        tx, ty = x + radius + 5, y
        label_draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0, 220), font=font)
        label_draw.text((tx, ty), label, fill=(255, 255, 255, 230), font=font)

    # 4. 결과물 저장
    filename = os.path.basename(image_path)
    output_filename = f"heatmap_{filename}"
    output_path = os.path.join(str(settings.outputs_dir), output_filename)
    
    out = Image.alpha_composite(base, overlay).convert("RGB")
    out.save(output_path)

    return output_path


def draw_points_on_image_normalized(
    image_path: str | Path,
    prediction: dict[str, Any],
    out_path: str | Path | None = None,
    *,
    collision_color: Tuple[int, int, int] = (255, 60, 60),
    bottleneck_color: Tuple[int, int, int] = (60, 120, 255),
) -> Any:
    """
    1단계 Vision 분석 JSON(정규화 좌표 x,y in [0,1])을 사용해 설계도 위에 마커를 그립니다.
    - 빨간 원: 충돌 위험 (collision_candidates)
    - 파란 원: 병목 (bottleneck_candidates)
    out_path가 있으면 파일로 저장하고 경로 반환. 없으면 BytesIO PNG 바이트 반환.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=max(14, w // 45))
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", size=max(14, w // 45))
        except Exception:
            font = ImageFont.load_default()

    def to_px(x_norm: float, y_norm: float) -> Tuple[int, int]:
        x = int(max(0, min(1, x_norm)) * w)
        y = int(max(0, min(1, y_norm)) * h)
        return x, y

    r = max(10, min(w, h) // 35)

    for p in prediction.get("collision_candidates", []):
        x, y = to_px(float(p["x"]), float(p["y"]))
        label = p.get("id", "C?")
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            outline=collision_color,
            width=4,
            fill=(*collision_color, 60),
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle(
            (x + r + 4, y - th // 2 - 2, x + r + 4 + tw + 6, y + th // 2 + 2),
            fill=(0, 0, 0, 140),
        )
        draw.text((x + r + 7, y - th // 2), label, font=font, fill=(255, 255, 255, 255))

    for p in prediction.get("bottleneck_candidates", []):
        x, y = to_px(float(p["x"]), float(p["y"]))
        label = p.get("id", "B?")
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            outline=bottleneck_color,
            width=4,
            fill=(*bottleneck_color, 60),
        )
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle(
            (x + r + 4, y - th // 2 - 2, x + r + 4 + tw + 6, y + th // 2 + 2),
            fill=(0, 0, 0, 140),
        )
        draw.text((x + r + 7, y - th // 2), label, font=font, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(img, overlay).convert("RGB")
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.save(out_path)
        return str(out_path)
    buf = BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    return buf