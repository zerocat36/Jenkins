from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


app = FastAPI(
    title="Factory Blueprint RAG Pipeline API",
    description="설계도+시나리오 분석(RAG/LLM) 및 히트맵/리포트/Isaac 파라미터 생성",
    version="0.1.0",
)

# CORS 설정: rob-java(8090), factory-robot-app(5174), localhost:80(nginx)
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://localhost:80",
    "http://127.0.0.1:80",
    "http://localhost:8090",
    "http://127.0.0.1:8090",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/v1")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)