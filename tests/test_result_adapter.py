"""
test_result_adapter.py — ResultAdapter 단위 테스트

실행:
    pytest tests/test_result_adapter.py -v

테스트 전략:
    - 실제 PaddleOCR 모델 없이 MockPred 객체를 사용
    - bbox 추출 우선순위(rec_boxes → rec_polys → dt_polys) 각각 검증
    - 최종 JSON 스키마 구조 검증
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.result_adapter import ResultAdapter


# ── Mock 객체 ─────────────────────────────────────────────────────────────────

class MockPred:
    """
    PaddleOCR pred 객체를 흉내내는 Mock 클래스.

    실제 모델 없이 다양한 to_dict() 반환값을 시뮬레이션한다.
    raw_dict에 원하는 데이터를 넣어 각 버전의 PaddleOCR을 재현할 수 있음.
    """

    def __init__(self, raw_dict: dict):
        self._raw = raw_dict

    def to_dict(self) -> dict:
        return self._raw


def make_image(tmp_path: Path, width: int = 100, height: int = 200) -> Path:
    """테스트용 더미 이미지 파일 생성. PIL로 실제 PNG 파일을 만들어 크기를 읽을 수 있게 함."""
    from PIL import Image as PILImage
    img_path = tmp_path / "test.jpg"
    PILImage.new("RGB", (width, height), color=(255, 255, 255)).save(str(img_path))
    return img_path


# ── 이미지 크기 읽기 테스트 ───────────────────────────────────────────────────

class TestGetImageSize:
    """_get_image_size(): 이미지 width/height 읽기"""

    def setup_method(self):
        self.adapter = ResultAdapter()

    def test_returns_correct_width_height(self, tmp_path):
        img_path = make_image(tmp_path, width=640, height=480)
        w, h = self.adapter._get_image_size(img_path)
        assert w == 640
        assert h == 480


# ── bbox 추출 우선순위 테스트 ─────────────────────────────────────────────────

class TestExtractBoxes:
    """_extract_boxes(): rec_boxes → rec_polys → dt_polys 순서로 추출"""

    def setup_method(self):
        self.adapter = ResultAdapter()

    def test_uses_rec_boxes_when_available(self):
        """rec_boxes가 있으면 AABB로 바로 사용"""
        raw = {"rec_boxes": [[10, 20, 110, 70]]}
        boxes = self.adapter._extract_boxes(raw)
        assert boxes == [(10, 20, 110, 70)]

    def test_falls_back_to_rec_polys(self):
        """rec_boxes가 없으면 rec_polys에서 변환"""
        poly = np.array([[100, 200], [250, 190], [255, 240], [105, 250]])
        raw = {"rec_polys": [poly]}
        boxes = self.adapter._extract_boxes(raw)
        # min: (100, 190), max: (255, 250)
        assert boxes == [(100, 190, 255, 250)]

    def test_falls_back_to_dt_polys(self):
        """rec_boxes, rec_polys 모두 없으면 dt_polys 사용"""
        poly = np.array([[0, 0], [50, 0], [50, 30], [0, 30]])
        raw = {"dt_polys": [poly]}
        boxes = self.adapter._extract_boxes(raw)
        assert boxes == [(0, 0, 50, 30)]

    def test_returns_empty_when_no_boxes(self):
        """박스 정보가 전혀 없으면 빈 리스트"""
        boxes = self.adapter._extract_boxes({})
        assert boxes == []

    def test_ignores_empty_rec_boxes(self):
        """rec_boxes 키는 있지만 빈 리스트면 rec_polys로 넘어가야 함"""
        poly = np.array([[0, 0], [100, 0], [100, 50], [0, 50]])
        raw = {"rec_boxes": [], "rec_polys": [poly]}
        boxes = self.adapter._extract_boxes(raw)
        # rec_boxes가 비어있으므로 rec_polys를 사용
        assert boxes == [(0, 0, 100, 50)]


# ── 폴리곤 → AABB 변환 테스트 ────────────────────────────────────────────────

class TestPolyToAabb:
    """_poly_to_aabb(): 폴리곤 좌표 → (x1, y1, x2, y2)"""

    def setup_method(self):
        self.adapter = ResultAdapter()

    def test_axis_aligned_rectangle(self):
        """회전 없는 직사각형 폴리곤"""
        poly = [[10, 20], [60, 20], [60, 50], [10, 50]]
        assert self.adapter._poly_to_aabb(poly) == (10, 20, 60, 50)

    def test_rotated_polygon(self):
        """기울어진 폴리곤 → 감싸는 AABB"""
        poly = [[100, 200], [250, 190], [255, 240], [105, 250]]
        x1, y1, x2, y2 = self.adapter._poly_to_aabb(poly)
        assert x1 == 100
        assert y1 == 190
        assert x2 == 255
        assert y2 == 250

    def test_returns_integers(self):
        """결과가 정수 타입인지 확인 (JSON 직렬화 시 float 방지)"""
        poly = [[0.5, 1.5], [10.5, 1.5], [10.5, 5.5], [0.5, 5.5]]
        result = self.adapter._poly_to_aabb(poly)
        assert all(isinstance(v, int) for v in result)


# ── label 생성 테스트 ─────────────────────────────────────────────────────────

class TestMakeLabel:
    """_make_label(): 텍스트 + 박스 → label dict"""

    def setup_method(self):
        self.adapter = ResultAdapter()

    def test_label_structure(self):
        """label dict의 키 구조 검증"""
        label = self.adapter._make_label(idx=0, text="안녕", box=(10, 20, 110, 70))
        assert set(label.keys()) == {"id", "index", "comment", "code", "mark"}

    def test_comment_equals_code(self):
        """comment와 code는 동일한 텍스트여야 함"""
        label = self.adapter._make_label(idx=0, text="테스트", box=(0, 0, 100, 50))
        assert label["comment"] == label["code"] == "테스트"

    def test_mark_values(self):
        """mark의 x, y, width, height, type 값 검증"""
        label = self.adapter._make_label(idx=1, text="텍스트", box=(10, 20, 110, 70))
        mark = label["mark"]
        assert mark["x"] == 10
        assert mark["y"] == 20
        assert mark["width"] == 100   # 110 - 10
        assert mark["height"] == 50   # 70 - 20
        assert mark["type"] == "RECT"

    def test_index_is_integer(self):
        """index는 정수여야 함 (문자열 아님)"""
        label = self.adapter._make_label(idx=3, text="x", box=(0, 0, 10, 10))
        assert isinstance(label["index"], int)
        assert label["index"] == 3

    def test_id_is_uuid_format(self):
        """id가 UUID4 형식인지 확인"""
        import re
        label = self.adapter._make_label(idx=0, text="x", box=(0, 0, 10, 10))
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, label["id"]), f"UUID4 형식이 아님: {label['id']}"

    def test_each_label_has_unique_id(self):
        """여러 label을 만들 때 ID가 모두 달라야 함"""
        ids = {self.adapter._make_label(i, "x", (0, 0, 10, 10))["id"] for i in range(10)}
        assert len(ids) == 10  # 10개 모두 고유해야 함


# ── convert() 전체 흐름 테스트 ────────────────────────────────────────────────

class TestConvert:
    """convert(): 전체 변환 흐름 통합 검증"""

    def setup_method(self):
        self.adapter = ResultAdapter()

    def test_convert_with_rec_boxes(self, tmp_path):
        """rec_boxes가 있는 pred를 변환"""
        img_path = make_image(tmp_path, width=2481, height=3509)
        pred = MockPred({
            "rec_texts": ["라", "마"],
            "rec_boxes": [
                [861, 393, 955, 523],    # x1,y1,x2,y2
                [991, 560, 1083, 705],
            ],
        })

        result = self.adapter.convert(pred, img_path)

        # 최상위 구조
        assert set(result.keys()) == {"info", "labels"}
        # info
        assert result["info"] == {"width": 2481, "height": 3509}
        # labels 수
        assert len(result["labels"]) == 2
        # 첫 번째 label 내용
        first = result["labels"][0]
        assert first["comment"] == "라"
        assert first["mark"]["x"] == 861
        assert first["mark"]["width"] == 94      # 955 - 861
        assert first["mark"]["height"] == 130    # 523 - 393

    def test_convert_with_rec_polys(self, tmp_path):
        """rec_polys 폴리곤을 AABB로 변환"""
        img_path = make_image(tmp_path)
        pred = MockPred({
            "rec_texts": ["안녕"],
            "rec_polys": [
                np.array([[100, 200], [300, 190], [305, 250], [105, 260]])
            ],
        })

        result = self.adapter.convert(pred, img_path)

        mark = result["labels"][0]["mark"]
        assert mark["x"] == 100
        assert mark["y"] == 190

    def test_convert_empty_pred(self, tmp_path):
        """인식 결과가 없는 경우 labels가 빈 리스트"""
        img_path = make_image(tmp_path)
        pred = MockPred({"rec_texts": [], "rec_boxes": []})

        result = self.adapter.convert(pred, img_path)

        assert result["labels"] == []

    def test_convert_pred_without_to_dict(self, tmp_path):
        """to_dict() 메서드가 없는 pred도 처리 (빈 결과)"""
        img_path = make_image(tmp_path)
        pred = "invalid_pred_object"  # to_dict() 없는 객체

        result = self.adapter.convert(pred, img_path)

        assert result["labels"] == []

    def test_index_is_sequential(self, tmp_path):
        """labels의 index가 0부터 순서대로 증가하는지"""
        img_path = make_image(tmp_path)
        pred = MockPred({
            "rec_texts": ["가", "나", "다"],
            "rec_boxes": [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ],
        })

        result = self.adapter.convert(pred, img_path)

        indices = [label["index"] for label in result["labels"]]
        assert indices == [0, 1, 2]
