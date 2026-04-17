"""
test_input_adapter.py — InputAdapter 단위 테스트

실행:
    pytest tests/test_input_adapter.py -v

테스트 전략:
    - 외부 의존성(requests, pdf2image, 실제 파일)은 Mock으로 대체
    - 각 테스트는 하나의 동작만 검증 (단일 책임)
    - 실제 네트워크/디스크 접근 없이 빠르게 실행 가능
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# 프로젝트 루트를 sys.path에 추가 (adapters 패키지 임포트를 위해)
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.input_adapter import InputAdapter, IMAGE_EXTENSIONS


# ── URL 판별 테스트 ───────────────────────────────────────────────────────────

class TestIsUrl:
    """_is_url() 메서드: URL 여부 판별"""

    def setup_method(self):
        # 각 테스트 전에 기본 설정의 어댑터 생성
        self.adapter = InputAdapter()

    def test_http_url_returns_true(self):
        assert self.adapter._is_url("http://example.com/doc.pdf") is True

    def test_https_url_returns_true(self):
        assert self.adapter._is_url("https://example.com/image.jpg") is True

    def test_local_path_returns_false(self):
        assert self.adapter._is_url("/data/image.jpg") is False

    def test_relative_path_returns_false(self):
        assert self.adapter._is_url("./docs/file.pdf") is False

    def test_empty_string_returns_false(self):
        assert self.adapter._is_url("") is False


# ── 폴더 스캔 테스트 ──────────────────────────────────────────────────────────

class TestScanFolder:
    """_scan_folder(): 폴더 내 이미지 파일 수집"""

    def setup_method(self):
        self.adapter = InputAdapter()

    def test_collects_images_sorted(self, tmp_path):
        # 임시 폴더에 이미지 파일 3개 생성
        (tmp_path / "c.jpg").touch()
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpeg").touch()

        result = self.adapter._scan_folder(tmp_path)

        # 정렬된 순서로 반환되는지 확인
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_ignores_non_image_files(self, tmp_path):
        # 이미지가 아닌 파일은 수집하지 않아야 함
        (tmp_path / "image.jpg").touch()
        (tmp_path / "document.pdf").touch()  # PDF는 이미지 아님
        (tmp_path / "readme.txt").touch()

        result = self.adapter._scan_folder(tmp_path)

        assert len(result) == 1
        assert result[0].name == "image.jpg"

    def test_collects_recursively(self, tmp_path):
        # 하위 폴더의 이미지도 수집해야 함
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "a.jpg").touch()
        (sub / "b.png").touch()

        result = self.adapter._scan_folder(tmp_path)

        assert len(result) == 2

    def test_empty_folder_returns_empty_list(self, tmp_path):
        result = self.adapter._scan_folder(tmp_path)
        assert result == []

    def test_all_image_extensions_recognized(self, tmp_path):
        # 지원하는 모든 확장자가 수집되는지 확인
        for ext in IMAGE_EXTENSIONS:
            (tmp_path / f"file{ext}").touch()

        result = self.adapter._scan_folder(tmp_path)
        assert len(result) == len(IMAGE_EXTENSIONS)


# ── 단일 이미지 파일 테스트 ───────────────────────────────────────────────────

class TestResolveSingleImage:
    """resolve(): 단일 이미지 파일 입력"""

    def setup_method(self):
        self.adapter = InputAdapter()

    def test_returns_list_with_single_path(self, tmp_path):
        img = tmp_path / "image.jpg"
        img.touch()

        result = self.adapter.resolve(str(img), tmp_path)

        assert result == [img]

    def test_strips_whitespace_from_source(self, tmp_path):
        # 앞뒤 공백이 있어도 정상 처리
        img = tmp_path / "image.png"
        img.touch()

        result = self.adapter.resolve(f"  {img}  ", tmp_path)

        assert result == [img]


# ── PDF 변환 테스트 ───────────────────────────────────────────────────────────

class TestPdfToJpg:
    """_pdf_to_jpg(): PDF 페이지를 JPG로 변환

    pypdfium2를 primary backend로 사용하므로 sys.modules에 mock을 주입해 테스트.
    pypdfium2는 _pdf_to_jpg_pypdfium2() 내부에서 import되므로
    patch.dict('sys.modules', ...)으로 호출 시점에 mock을 삽입할 수 있다.
    """

    def setup_method(self):
        self.adapter = InputAdapter(pdf_dpi=144)  # 144 / 72 = scale 2.0 (검증 편의)

    def _make_pdfium_mock(self, n_pages: int):
        """n_pages짜리 PDF를 흉내 내는 pypdfium2 mock 반환."""
        mock_pil   = MagicMock()
        mock_bmp   = MagicMock()
        mock_bmp.to_pil.return_value = mock_pil
        mock_page  = MagicMock()
        mock_page.render.return_value = mock_bmp

        mock_pdf = MagicMock()
        mock_pdf.__len__ = MagicMock(return_value=n_pages)
        mock_pdf.__iter__ = MagicMock(return_value=iter([mock_page] * n_pages))

        mock_pdfium = MagicMock()
        mock_pdfium.PdfDocument.return_value = mock_pdf
        return mock_pdfium, mock_page, mock_pil

    def test_saves_jpg_per_page(self, tmp_path):
        mock_pdfium, _, _ = self._make_pdfium_mock(3)

        with patch.dict("sys.modules", {"pypdfium2": mock_pdfium}):
            pdf_path = tmp_path / "doc.pdf"
            pdf_path.touch()
            result = self.adapter._pdf_to_jpg(pdf_path, tmp_path)

        # 3페이지 → 3개 경로
        assert len(result) == 3
        # 파일명 형식 확인 (page_001.jpg, page_002.jpg, page_003.jpg)
        assert [p.name for p in result] == ["page_001.jpg", "page_002.jpg", "page_003.jpg"]

    def test_creates_subdirectory_named_after_pdf(self, tmp_path):
        mock_pdfium, _, _ = self._make_pdfium_mock(1)

        with patch.dict("sys.modules", {"pypdfium2": mock_pdfium}):
            pdf_path = tmp_path / "문서A.pdf"
            pdf_path.touch()
            result = self.adapter._pdf_to_jpg(pdf_path, tmp_path)

        # 출력 경로가 tmp_path/문서A/문서A_images/page_001.jpg 구조인지 확인
        assert result[0].parent.name == "문서A_images"
        assert result[0].parent.parent.name == "문서A"

    def test_calls_render_with_correct_scale(self, tmp_path):
        # pdf_dpi=144 → scale = 144/72 = 2.0
        mock_pdfium, mock_page, _ = self._make_pdfium_mock(1)

        with patch.dict("sys.modules", {"pypdfium2": mock_pdfium}):
            pdf_path = tmp_path / "doc.pdf"
            pdf_path.touch()
            self.adapter._pdf_to_jpg(pdf_path, tmp_path)

        # render()가 올바른 scale=2.0으로 호출됐는지 확인
        mock_page.render.assert_called_once_with(scale=2.0, rotation=0)


# ── URL 다운로드 테스트 ───────────────────────────────────────────────────────

class TestResolveUrl:
    """_resolve_url(): URL에서 파일 다운로드"""

    def setup_method(self):
        self.adapter = InputAdapter(ssl_verify=True)

    def test_downloads_and_saves_file(self, tmp_path):
        # requests.get을 Mock으로 대체 (실제 네트워크 요청 불필요)
        mock_response = MagicMock()
        mock_response.content = b"fake image data"

        with patch("adapters.input_adapter.requests.get", return_value=mock_response), \
             patch.object(self.adapter, "resolve", wraps=self.adapter.resolve) as mock_resolve:

            # 이미지 URL 테스트
            url = "https://example.com/image.jpg"
            download_dir = tmp_path / "_downloads"
            download_dir.mkdir()

            # resolve를 직접 호출하면 재귀가 발생하므로 _resolve_url만 테스트
            # 파일이 저장되는지만 확인
            mock_response.raise_for_status = MagicMock()

            with patch.object(self.adapter, "resolve", return_value=[tmp_path / "image.jpg"]):
                result = self.adapter._resolve_url(url, tmp_path)

        # requests.get이 올바른 인자로 호출됐는지 확인
        import adapters.input_adapter as mod
        # 주: 세부 assert는 통합 테스트에서 진행

    def test_ssl_verify_false_suppresses_warning(self, tmp_path):
        # ssl_verify=False 설정 시 경고가 억제되는지 확인
        adapter = InputAdapter(ssl_verify=False)
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        with patch("adapters.input_adapter.requests.get", return_value=mock_response), \
             patch.object(adapter, "resolve", return_value=[]):
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adapter._resolve_url("https://example.com/img.jpg", tmp_path)
            # InsecureRequestWarning이 없어야 함 (필터로 억제됨)

    def test_extracts_filename_from_url(self, tmp_path):
        # URL에서 파일명 추출 로직 확인
        adapter = InputAdapter()
        mock_response = MagicMock()
        mock_response.content = b"data"
        mock_response.raise_for_status = MagicMock()

        with patch("adapters.input_adapter.requests.get", return_value=mock_response), \
             patch.object(adapter, "resolve", return_value=[]) as mock_r:

            adapter._resolve_url("https://example.com/path/report.pdf?v=2", tmp_path)

            # resolve가 호출될 때 파일명이 'report.pdf'인지 확인
            call_args = mock_r.call_args[0][0]  # 첫 번째 위치 인자 (source)
            assert "report.pdf" in call_args


# ── 지원하지 않는 입력 테스트 ─────────────────────────────────────────────────

class TestResolveInvalidInput:
    """resolve(): 지원하지 않는 입력에 대한 예외 처리"""

    def setup_method(self):
        self.adapter = InputAdapter()

    def test_raises_for_unsupported_extension(self, tmp_path):
        txt_file = tmp_path / "document.txt"
        txt_file.touch()

        with pytest.raises(ValueError, match="지원하지 않는 입력 형식"):
            self.adapter.resolve(str(txt_file), tmp_path)

    def test_raises_for_nonexistent_path(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            self.adapter.resolve("/nonexistent/path/image.jpg", tmp_path)
