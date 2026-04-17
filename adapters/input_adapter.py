"""
input_adapter.py — 입력 정규화 어댑터

역할:
    다양한 입력 형태(URL, PDF, 이미지 파일, 폴더)를
    OCR 모델이 처리할 수 있는 이미지 파일 경로 목록으로 통일한다.

단일 책임 원칙:
    '입력을 이미지 경로 목록으로 변환'만 담당.
    OCR 추론이나 결과 저장은 하지 않는다.

교체 포인트:
    다른 입력 소스(S3, GCS, FTP 등)를 추가할 때 이 파일만 수정하면 된다.
    InputAdapterProtocol을 만족하는 새 클래스를 만들어 Pipeline에 주입하면 됨.
"""

import warnings
from pathlib import Path
from typing import Protocol

import requests


# 처리할 수 있는 이미지 확장자 집합
# set을 쓰는 이유: 'in' 연산이 list보다 O(1)로 빠름
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── 인터페이스 정의 ───────────────────────────────────────────────────────────

class InputAdapterProtocol(Protocol):
    """
    InputAdapter 인터페이스 (덕 타이핑 방식의 Protocol).

    추상 클래스(ABC) 대신 Protocol을 사용하는 이유:
    - 상속 없이 resolve() 메서드만 있으면 이 타입을 만족함
    - 테스트용 Mock 클래스를 상속 없이 자유롭게 만들 수 있음
    - Pipeline이 구체 구현체가 아닌 인터페이스에 의존하게 됨 (DIP 원칙)
    """

    def resolve(self, source: str | list[str], jpg_dir: Path) -> list[Path]:
        """
        source  : 입력 경로·URL (문자열) 또는 경로 문자열 리스트
        jpg_dir : PDF 변환 JPG / URL 다운로드 파일 저장 루트 폴더
        반환    : 정렬된 이미지 파일 경로 목록
        """
        ...


# ── 구현체 ────────────────────────────────────────────────────────────────────

class InputAdapter:
    """
    실제 InputAdapter 구현체.

    의존성 주입(DI):
    - pdf_dpi, ssl_verify, download_timeout을 생성자에서 주입받음
    - 테스트 시 다른 값을 주입해 동작을 제어할 수 있음
    """

    def __init__(
        self,
        pdf_dpi: int = 200,
        jpg_quality: int = 75,
        ssl_verify: bool = True,
        download_timeout: int = 30,
    ):
        """
        pdf_dpi          : PDF 페이지를 JPG로 변환할 해상도
                           높을수록 품질↑ 속도↓ (OCR엔 150~300 DPI 권장)
        jpg_quality      : JPEG 저장 품질 (1~95)
                           75: PIL 기본값(균형) / 95: 고품질(파일 크기↑, 아티팩트↓)
        ssl_verify       : HTTPS 요청 시 SSL 인증서 검증 여부
                           False → 자체서명 인증서도 허용 (내부망 서버 등)
        download_timeout : URL 다운로드 최대 대기 시간(초)
        """
        self.pdf_dpi = pdf_dpi
        self.jpg_quality = jpg_quality
        self.ssl_verify = ssl_verify
        self.download_timeout = download_timeout

    def resolve(self, source: str | list[str], jpg_dir: Path) -> list[Path]:
        """
        입력 source를 이미지 경로 목록으로 변환하는 진입점.

        처리 분기 순서:
        0. list[str]       → 각 항목을 재귀 처리 후 합산
        1. URL (http/https) → 파일 다운로드 후 재귀 처리
        2. .pdf 파일        → 페이지별 JPG 변환
        3. 이미지 파일      → [해당 경로] 그대로 반환
        4. 폴더             → 하위 이미지 전체 수집
        """
        if isinstance(source, list):
            return self._resolve_list(source, jpg_dir)

        source = source.strip()

        if self._is_url(source):
            return self._resolve_url(source, jpg_dir)

        path = Path(source)

        if path.suffix.lower() == ".pdf":
            return self._pdf_to_jpg(path, jpg_dir)

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            if not path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
            return [path]

        if path.is_dir():
            return self._scan_folder(path)

        raise ValueError(
            f"지원하지 않는 입력 형식: {source!r}\n"
            f"지원 형식: PDF, 이미지 파일({', '.join(IMAGE_EXTENSIONS)}), 폴더, URL, list[str]"
        )

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _resolve_list(self, sources: list[str], jpg_dir: Path) -> list[Path]:
        """경로 문자열 리스트를 받아 각 항목을 resolve() 재귀 처리 후 합산."""
        result = []
        for s in sources:
            result.extend(self.resolve(s, jpg_dir))
        return result

    def _is_url(self, source: str) -> bool:
        """http:// 또는 https:// 로 시작하면 URL로 판단."""
        return source.startswith(("http://", "https://"))

    def _resolve_url(self, url: str, jpg_dir: Path) -> list[Path]:
        """
        URL에서 파일을 다운로드하고 타입에 맞게 처리.

        저장 경로: jpg_dir/_downloads/원본파일명
        다운로드 후 resolve()를 재귀 호출해 PDF/이미지 처리를 위임.
        재귀 호출 이유: URL이 PDF일 수도, 이미지일 수도 있어
                       각 경우의 처리 로직을 중복 구현하지 않기 위함.
        """
        if not self.ssl_verify:
            # ssl_verify=False일 때 발생하는 InsecureRequestWarning 숨기기
            # 사용자가 이미 의도적으로 SSL 우회를 선택했으므로 경고가 불필요함
            warnings.filterwarnings("ignore", message="Unverified HTTPS request")

        # URL 마지막 세그먼트에서 파일명 추출 (쿼리스트링 제거)
        # 예) https://example.com/docs/report.pdf?v=2 → "report.pdf"
        filename = url.split("/")[-1].split("?")[0] or "downloaded_file"

        download_dir = jpg_dir / "_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)

        # stream=True: 큰 파일을 메모리에 한 번에 올리지 않고 청크 단위로 받음
        # User-Agent: 일부 서버가 봇 차단을 위해 브라우저 헤더를 요구함
        response = requests.get(
            url,
            verify=self.ssl_verify,
            timeout=self.download_timeout,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()   # 4xx/5xx 응답이면 예외 발생

        # 파일명에 확장자가 없으면 Content-Type으로 보정
        # 예) URL이 /image/jpeg 처럼 확장자 없는 경우
        if "." not in Path(filename).suffix:
            content_type = response.headers.get("content-type", "")
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png",
                "image/bmp": ".bmp", "image/tiff": ".tif",
                "image/webp": ".webp", "application/pdf": ".pdf",
            }
            for mime, ext in ext_map.items():
                if mime in content_type:
                    filename += ext
                    break

        save_path = download_dir / filename
        save_path.write_bytes(response.content)

        # 다운로드된 파일을 일반 파일처럼 처리 (재귀)
        return self.resolve(str(save_path), jpg_dir)

    def _pdf_to_jpg(self, pdf_path: Path, jpg_dir: Path) -> list[Path]:
        """
        PDF 각 페이지를 JPG 파일로 변환 후 저장.

        저장 경로 예시:
            입력: /data/문서A.pdf
            출력: jpg_dir/문서A/문서A_images/page_001.jpg
                  jpg_dir/문서A/문서A_images/page_002.jpg
                  ...
          JSON은 Pipeline._save_json()이 jpg_dir/문서A/문서A_labels/ 에 저장.

        pypdfium2 우선 사용 (poppler 불필요, 순수 Python 패키지).
        없으면 pdf2image(poppler 필요)로 fallback.
        """
        out_dir = jpg_dir / pdf_path.stem / f"{pdf_path.stem}_images"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            return self._pdf_to_jpg_pypdfium2(pdf_path, out_dir)
        except ImportError:
            return self._pdf_to_jpg_pdf2image(pdf_path, out_dir)

    def _pdf_to_jpg_pypdfium2(self, pdf_path: Path, out_dir: Path) -> list[Path]:
        """pypdfium2 백엔드 — poppler 없이 동작."""
        import pypdfium2 as pdfium
        from tqdm.auto import tqdm as tqdm_pdf

        # DPI → scale 변환: pypdfium2는 72 DPI 기준 scale 배수를 받음
        scale = self.pdf_dpi / 72

        pdf = pdfium.PdfDocument(str(pdf_path))
        total = len(pdf)
        saved_paths = []

        for i, page in enumerate(tqdm_pdf(pdf, desc="PDF→JPG", total=total, dynamic_ncols=True), start=1):
            bitmap = page.render(scale=scale, rotation=0)
            pil_img = bitmap.to_pil()
            # page_001.jpg … 형식: 0패딩으로 정렬 재현성 보장
            jpg_path = out_dir / f"page_{i:03d}.jpg"
            pil_img.save(str(jpg_path), "JPEG", quality=self.jpg_quality)
            saved_paths.append(jpg_path)

        return saved_paths

    def _pdf_to_jpg_pdf2image(self, pdf_path: Path, out_dir: Path) -> list[Path]:
        """pdf2image 백엔드 — poppler-utils 필요."""
        from pdf2image import convert_from_path

        pages = convert_from_path(str(pdf_path), dpi=self.pdf_dpi)
        saved_paths = []
        for i, page_img in enumerate(pages, start=1):
            jpg_path = out_dir / f"page_{i:03d}.jpg"
            page_img.save(str(jpg_path), "JPEG")
            saved_paths.append(jpg_path)
        return saved_paths

    def _scan_folder(self, folder: Path) -> list[Path]:
        """
        폴더 내 이미지 파일을 재귀적으로 수집.

        sorted() 사용 이유:
            OS/파일시스템마다 rglob 반환 순서가 다를 수 있음.
            정렬로 실행 순서를 항상 동일하게 보장 → 결과 재현성 확보.
        """
        return sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
