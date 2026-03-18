# Dockerfile
FROM python:3.11-slim

# 기본 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 설치
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 앱 코드 복사
COPY app /app/app

# 데이터 디렉터리 생성 (앱이 런타임에 uploads/outputs/vector_db 생성)
RUN mkdir -p /app/data

# 서비스 포트
EXPOSE 8003

# 실행
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]