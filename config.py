"""
config.py — 전역 설정값 모음

설계 원칙:
- dataclass로 타입 힌트와 기본값을 한 곳에서 선언
- main.py(CLI)와 api.py(FastAPI) 양쪽에서 동일하게 사용
- 테스트 시 다른 값의 Config 인스턴스를 만들어 Pipeline에 주입 가능
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ── OCR 모델 설정 ─────────────────────────────────────────
    lang: str      = "korean"   # 인식 언어 (korean / en / chinese_cht 등)
    device: str    = "gpu:0"    # 추론 장치: "gpu:0", "gpu:1", "cpu"
    precision: str = "fp32"     # 연산 정밀도: "fp16"(빠름) / "fp32"(안정)

    # ── 추론 설정 ─────────────────────────────────────────────
    batch_size: int = 1         # 한 번에 처리할 이미지 수 (기본 1장 = 단일 추론)

    # ── PDF 변환 설정 ─────────────────────────────────────────
    pdf_dpi: int     = 200      # PDF → JPG 변환 해상도
                                # 150: 속도 우선 / 200: 균형점 / 300: 품질 우선
    jpg_quality: int = 75       # JPEG 저장 품질 (1~95)
                                # 75: PIL 기본값(균형) / 95: 고품질(파일 크기↑)

    # ── 문서 방향 설정 ────────────────────────────────────────
    # PDF 변환 이미지처럼 이미 정립된 소스는 모두 False(기본값)로 속도를 높임.
    # 스캔본·사진 등 방향이 불확실한 이미지 처리 시 필요한 항목만 True로 설정.
    use_doc_orientation_classify: bool = False  # 문서 회전 감지·보정 (0/90/180/270°)
    use_textline_orientation:     bool = False  # 텍스트라인 방향 감지 (가로↔세로)
    use_doc_unwarping:            bool = False  # 문서 왜곡(구김·굴곡) 보정 — 가장 느림

    # ── 네트워크 설정 ─────────────────────────────────────────
    ssl_verify: bool    = True  # False 시 자체서명 인증서도 허용 (SSL 우회)
    download_timeout: int = 30  # URL 다운로드 최대 대기 시간 (초)
