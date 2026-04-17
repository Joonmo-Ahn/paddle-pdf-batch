"""
id_gen.py — 고유 ID 생성 유틸리티

역할: JSON 각 label에 붙이는 고유 식별자를 생성.
      UUID4 사용 이유: 완전 무작위라 충돌 확률이 사실상 0이고,
      외부 시스템(DB, 프론트엔드)과 키 충돌 없이 연동 가능.
"""

import uuid


def random_id() -> str:
    """UUID4 형식의 고유 ID 문자열 반환.

    예) "f3d4a0b4-725f-4d9d-961e-522c390072f9"
    """
    return str(uuid.uuid4())
