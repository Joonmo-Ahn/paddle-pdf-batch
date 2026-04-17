"""
main.py — CLI 진입점

역할:
    커맨드라인 인자를 파싱하고, Config를 구성한 뒤 파이프라인을 실행한다.
    파이프라인 로직은 pipeline.py에, 설정 기본값은 config.py에 위임.

실행 예:
    # 단일 이미지 (기본: batch_size=1)
    python main.py /data/image.jpg --output ./results

    # PDF 파일, 배치 추론
    python main.py /data/문서A.pdf --output ./results --batch-size 8

    # 폴더 전체, GPU 추론
    python main.py /data/images/ --output ./results --batch-size 16

    # URL 입력 (SSL 인증서 검증 비활성화)
    python main.py https://example.com/doc.pdf --output ./results --no-ssl-verify

    # CPU 추론, 고해상도 PDF 변환
    python main.py /data/scan.pdf --output ./results --device cpu --dpi 300
"""

import argparse
from pathlib import Path

from config import Config
from pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    """커맨드라인 인자를 정의하고 파싱한다."""
    parser = argparse.ArgumentParser(
        description="PaddleOCR 배치 추론 파이프라인 (PDF / 이미지 / 폴더 / URL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # 모듈 docstring을 --help 하단에 표시
    )

    # 위치 인자: 입력 경로 또는 URL (필수)
    parser.add_argument(
        "source",
        help="입력 경로 또는 URL (.pdf / 이미지 파일 / 폴더 / https://...)",
    )

    # 옵션 인자
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="JSON 출력 폴더 경로",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="배치 크기 (기본: 1 = 단일 추론)",
    )
    parser.add_argument(
        "--device",
        default="gpu:0",
        help="추론 장치 (기본: gpu:0 / 예: gpu:1, cpu)",
    )
    parser.add_argument(
        "--precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="연산 정밀도 (기본: fp32 / fp16은 속도 빠르나 정밀도↓)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PDF → JPG 변환 해상도 (기본: 200 DPI)",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=75,
        help="JPEG 저장 품질 1~95 (기본: 75 / 고품질: 95)",
    )
    parser.add_argument(
        "--no-ssl-verify",
        action="store_true",        # 플래그 지정 시 True, 미지정 시 False
        help="SSL 인증서 검증 비활성화 (자체서명 인증서 서버 접근 시 사용)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # argparse 인자로 Config 값을 덮어써서 최종 설정 구성
    config = Config(
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        pdf_dpi=args.dpi,
        jpg_quality=args.jpg_quality,
        ssl_verify=not args.no_ssl_verify,  # --no-ssl-verify 지정 시 False
    )

    # 파이프라인 조립 (PaddleOCR 모델 초기화 포함 — 수 초 소요)
    pipeline = build_pipeline(config)

    # 파이프라인 실행
    pipeline.run(
        source=args.source,
        output_dir=Path(args.output),
        batch_size=config.batch_size,
    )


if __name__ == "__main__":
    main()
