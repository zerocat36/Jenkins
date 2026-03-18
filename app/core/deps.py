"""
세션 인증 의존성 — JWT 로컬 검증.
factory-backend와 동일한 JWT_SECRET을 공유해 HTTP 요청 없이 검증합니다.
"""
from __future__ import annotations

import os
from typing import Optional

import jwt
from fastapi import Cookie, HTTPException

JWT_SECRET = os.getenv("JWT_SECRET", "factory-robot-super-secret-key-change-in-prod!!")
JWT_ALGORITHM = "HS256"


async def verify_session(
    factory_robot_sid: Optional[str] = Cookie(default=None),
) -> str:
    if not factory_robot_sid:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    try:
        payload = jwt.decode(factory_robot_sid, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise ValueError("sub missing")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="세션이 만료되었습니다. 다시 로그인해 주세요.")
    except Exception:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
