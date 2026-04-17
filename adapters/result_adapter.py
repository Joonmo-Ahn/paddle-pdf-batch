"""
result_adapter.py — 추론 결과 변환 어댑터

역할:
    PaddleOCR의 raw 예측 결과를 내부 JSON 스키마로 변환한다.

단일 책임 원칙:
    '결과 변환'만 담당. 모델 추론이나 파일 저장은 하지 않는다.

교체 포인트:
    다른 OCR 모델(EasyOCR, Tesseract 등)로 바꿀 때 이 파일만 교체하면 된다.
    ResultAdapterProtocol을 만족하는 새 클래스를 만들어 Pipeline에 주입하면 됨.

출력 JSON 스키마:
    {
        "info": {"width": 2481, "height": 3509},
        "labels": [
            {
                "id": "<uuid4>",
                "index": 0,
                "comment": "인식된 텍스트",
                "code": "인식된 텍스트",
                "mark": {"x": 100, "y": 200, "width": 300, "height": 50, "type": "RECT"}
            },
            ...
        ]
    }
"""

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from utils.id_gen import random_id


# ── 인터페이스 정의 ───────────────────────────────────────────────────────────

class ResultAdapterProtocol(Protocol):
    """
    ResultAdapter 인터페이스.
    convert() 하나만 구현하면 이 타입을 만족함.
    """

    def convert(self, pred, image_path: Path) -> dict:
        """
        pred       : OCR 모델의 예측 결과 객체
        image_path : 이미지 파일 경로 (이미지 크기 읽기에 사용)
        반환       : 내부 JSON 스키마 dict
        """
        ...


# ── 구현체 ────────────────────────────────────────────────────────────────────

class ResultAdapter:
    """
    PaddleOCR 예측 결과를 내부 JSON 스키마로 변환하는 어댑터.

    상태(state)가 없는 순수 변환기이므로 멤버 변수가 없음.
    의존성 주입이 필요한 설정이 생기면 __init__에 추가하면 됨.
    """

    def convert(self, pred, image_path: Path) -> dict:
        """
        변환 진입점. PaddleOCR pred 객체 하나를 받아 dict를 반환.

        실행 순서:
        1. 이미지 크기 읽기 (info 섹션에 사용)
        2. pred.to_dict()로 raw 데이터 추출
        3. 텍스트 + 박스 좌표 추출
        4. label 목록 생성
        5. 최종 dict 반환
        """
        width, height = self._get_image_size(image_path)

        # PaddleOCR 버전별 반환 타입 대응:
        #   - 구버전: pred.to_dict() 메서드 보유
        #   - 신버전(OCRResult): dict 서브클래스로 직접 접근 가능 (keys(), get() 등 지원)
        if hasattr(pred, "to_dict"):
            raw: dict = pred.to_dict()
        elif hasattr(pred, "keys"):
            raw: dict = dict(pred)   # OCRResult(dict 서브클래스) → 일반 dict로 변환
        else:
            raw: dict = {}

        texts = raw.get("rec_texts", [])
        boxes = self._extract_boxes(raw)

        # 텍스트와 박스가 1:1 대응. zip이 짧은 쪽 기준으로 끊어줌.
        labels = [
            self._make_label(idx, text, box)
            for idx, (text, box) in enumerate(zip(texts, boxes))
        ]

        return {
            "info": {"width": width, "height": height},
            "labels": labels,
        }

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _get_image_size(self, image_path: Path) -> tuple[int, int]:
        """
        PIL로 이미지 크기(width, height)만 빠르게 읽는다.

        PIL Image.open() + .size 사용 이유:
        - cv2.imread()는 픽셀 데이터 전체를 메모리에 로드 → 느리고 무거움
        - PIL은 파일 헤더만 파싱해 width/height를 반환 → 가볍고 빠름
        - with 구문으로 파일 핸들을 자동으로 닫음 (리소스 누수 방지)
        """
        with Image.open(image_path) as img:
            return img.size  # (width, height) 튜플

    def _extract_boxes(self, raw: dict) -> list[tuple[int, int, int, int]]:
        """
        PaddleOCR 버전에 따라 다른 키에서 AABB 박스를 추출.

        AABB(Axis-Aligned Bounding Box): 회전 없는 직사각형 (x1, y1, x2, y2)

        시도 순서 (우선순위):
        1. rec_boxes  : 일부 버전에서 직접 AABB [x1, y1, x2, y2] 제공
        2. rec_polys  : 대부분 버전에서 제공하는 폴리곤 좌표 → AABB 변환
        3. dt_polys   : 탐지 전용 폴리곤 (rec_polys 없을 때 마지막 수단)
        4. 없으면 []  : 빈 리스트 반환 (labels도 비어있게 됨)
        """
        # 케이스 1: rec_boxes가 있고 비어있지 않으면 그대로 사용
        rec_boxes = raw.get("rec_boxes")
        if rec_boxes is not None and len(rec_boxes) > 0:
            return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in rec_boxes]

        # 케이스 2: rec_polys → AABB 변환
        rec_polys = raw.get("rec_polys")
        if rec_polys is not None and len(rec_polys) > 0:
            return [self._poly_to_aabb(poly) for poly in rec_polys]

        # 케이스 3: dt_polys → AABB 변환 (마지막 수단)
        dt_polys = raw.get("dt_polys", [])
        return [self._poly_to_aabb(poly) for poly in dt_polys]

    def _poly_to_aabb(self, poly) -> tuple[int, int, int, int]:
        """
        폴리곤 좌표 배열을 AABB (x1, y1, x2, y2)로 변환.

        폴리곤이 필요한 이유:
            OCR 모델은 기울어진 텍스트를 4점 폴리곤으로 인식함.
            하지만 우리 JSON 스키마는 직사각형(RECT)만 사용하므로
            폴리곤을 감싸는 최소 직사각형(AABB)으로 변환이 필요.

        변환 예시:
            입력 폴리곤: [[100,200], [250,190], [255,240], [105,250]]
            x 범위: 100~255  →  x1=100, x2=255
            y 범위: 190~250  →  y1=190, y2=250
            AABB: (100, 190, 255, 250)

        numpy min/max 사용 이유:
            점이 4개를 초과하는 폴리곤도 동일하게 처리 가능.
            axis=0은 "각 컬럼(x, y)에 대해 전체 행의 min/max"를 의미.
        """
        arr = np.array(poly)               # shape: (N, 2) — N개 점, (x, y)
        x1, y1 = arr.min(axis=0).tolist() # 모든 점 중 최솟값
        x2, y2 = arr.max(axis=0).tolist() # 모든 점 중 최댓값
        return (int(x1), int(y1), int(x2), int(y2))

    def _make_label(self, idx: int, text: str, box: tuple) -> dict:
        """
        텍스트 하나와 박스 하나를 받아 label dict를 생성.

        mark 좌표 변환:
            AABB (x1, y1, x2, y2) → {x, y, width, height}
            x      = x1          (좌상단 X 좌표)
            y      = y1          (좌상단 Y 좌표)
            width  = x2 - x1     (가로 길이)
            height = y2 - y1     (세로 길이)

        code == comment:
            현재 스키마에서는 인식 텍스트를 두 필드에 동일하게 저장.
            향후 code에 별도 분류/교정 로직이 생기면 이 메서드만 수정.
        """
        x1, y1, x2, y2 = box
        return {
            "id": random_id(),       # 각 label의 고유 식별자 (UUID4)
            "index": idx,            # 0부터 시작하는 순서 번호
            "comment": text,         # OCR 인식 텍스트
            "code": text,            # comment와 동일 (현재 스키마 기준)
            "mark": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "type": "RECT",      # 항상 직사각형 (현재 스키마 고정값)
            },
        }
