"""
test_pipeline.py — Pipeline 통합 테스트

실행:
    pytest tests/test_pipeline.py -v

테스트 전략:
    - 실제 PaddleOCR 모델 대신 MockModel 주입 (DI 패턴 덕분에 가능)
    - 실제 어댑터 인스턴스를 사용해 파이프라인 전체 흐름 검증
    - JSON 파일이 올바르게 생성됐는지 파일 시스템에서 확인
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.input_adapter import InputAdapter
from adapters.result_adapter import ResultAdapter
from pipeline import Pipeline


# ── Mock 객체 ─────────────────────────────────────────────────────────────────

class MockPred:
    """PaddleOCR pred 객체를 흉내내는 Mock."""

    def __init__(self, texts: list[str], boxes: list):
        self._data = {"rec_texts": texts, "rec_boxes": boxes}

    def to_dict(self) -> dict:
        return self._data


class MockModel:
    """
    PaddleOCR 모델을 흉내내는 Mock.

    predict()가 호출되면 이미지 수만큼 MockPred를 반환.
    실제 GPU 없이 파이프라인 흐름을 테스트할 수 있음.
    """

    def __init__(self, texts: list[str] = None, boxes: list = None):
        # 기본값: 텍스트 한 개, 박스 한 개
        self.texts = texts or ["테스트"]
        self.boxes = boxes or [[10, 20, 110, 70]]

    def predict(self, image_paths: list[str]):
        """이미지 경로 개수만큼 MockPred를 yield."""
        for _ in image_paths:
            yield MockPred(self.texts, self.boxes)


def make_test_image(path: Path, width: int = 200, height: int = 300):
    """테스트용 실제 이미지 파일 생성."""
    PILImage.new("RGB", (width, height), color=(255, 255, 255)).save(str(path))


# ── 파이프라인 기본 테스트 ────────────────────────────────────────────────────

class TestPipelineRun:
    """Pipeline.run(): 전체 파이프라인 실행"""

    def setup_method(self):
        # 각 테스트에서 공통으로 사용할 파이프라인 (Mock 모델 주입)
        self.pipeline = Pipeline(
            model=MockModel(),
            input_adapter=InputAdapter(),
            result_adapter=ResultAdapter(),
        )

    def test_single_image_creates_json(self, tmp_path):
        """단일 이미지 입력 → JSON 1개 생성"""
        img_path = tmp_path / "page_001.jpg"
        make_test_image(img_path)

        output_dir = tmp_path / "output"
        saved = self.pipeline.run(str(img_path), output_dir, batch_size=1)

        assert len(saved) == 1
        assert saved[0].suffix == ".json"
        assert saved[0].exists()

    def test_json_has_correct_schema(self, tmp_path):
        """생성된 JSON이 내부 스키마(info + labels)를 갖추는지"""
        img_path = tmp_path / "image.jpg"
        make_test_image(img_path, width=640, height=480)

        output_dir = tmp_path / "output"
        saved = self.pipeline.run(str(img_path), output_dir)

        data = json.loads(saved[0].read_text(encoding="utf-8"))
        # 최상위 키
        assert "info" in data
        assert "labels" in data
        # info 값
        assert data["info"]["width"] == 640
        assert data["info"]["height"] == 480

    def test_batch_processing_creates_multiple_jsons(self, tmp_path):
        """폴더 입력 + batch_size=2 → 이미지 수만큼 JSON 생성"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        # 이미지 5장 생성
        for i in range(5):
            make_test_image(images_dir / f"img_{i:02d}.jpg")

        output_dir = tmp_path / "output"
        saved = self.pipeline.run(str(images_dir), output_dir, batch_size=2)

        assert len(saved) == 5

    def test_returns_empty_for_no_images(self, tmp_path):
        """이미지 없는 폴더 입력 → 빈 리스트 반환"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        output_dir = tmp_path / "output"
        result = self.pipeline.run(str(empty_dir), output_dir)

        assert result == []

    def test_output_dir_created_automatically(self, tmp_path):
        """output_dir가 없어도 자동 생성"""
        img_path = tmp_path / "image.jpg"
        make_test_image(img_path)

        output_dir = tmp_path / "new" / "nested" / "output"
        assert not output_dir.exists()

        self.pipeline.run(str(img_path), output_dir)

        assert output_dir.exists()

    def test_json_label_content(self, tmp_path):
        """JSON labels의 실제 내용 검증"""
        pipeline = Pipeline(
            model=MockModel(texts=["안녕", "하세요"], boxes=[[0, 0, 50, 30], [60, 40, 150, 80]]),
            input_adapter=InputAdapter(),
            result_adapter=ResultAdapter(),
        )

        img_path = tmp_path / "image.jpg"
        make_test_image(img_path)

        output_dir = tmp_path / "output"
        saved = pipeline.run(str(img_path), output_dir)

        data = json.loads(saved[0].read_text(encoding="utf-8"))
        labels = data["labels"]

        assert len(labels) == 2
        assert labels[0]["comment"] == "안녕"
        assert labels[1]["comment"] == "하세요"
        assert labels[0]["index"] == 0
        assert labels[1]["index"] == 1
        assert labels[0]["mark"]["type"] == "RECT"


# ── 의존성 주입 테스트 ────────────────────────────────────────────────────────

class TestDependencyInjection:
    """DI 패턴: Mock 어댑터로 각 구성요소 독립 테스트"""

    def test_custom_result_adapter_is_used(self, tmp_path):
        """커스텀 ResultAdapter를 주입하면 해당 어댑터가 호출됨"""

        class CustomResultAdapter:
            """항상 고정된 결과를 반환하는 테스트용 어댑터"""
            def convert(self, pred, image_path):
                return {"info": {"width": 0, "height": 0}, "labels": [{"custom": True}]}

        pipeline = Pipeline(
            model=MockModel(),
            input_adapter=InputAdapter(),
            result_adapter=CustomResultAdapter(),
        )

        img_path = tmp_path / "image.jpg"
        make_test_image(img_path)

        saved = pipeline.run(str(img_path), tmp_path / "output")
        data = json.loads(saved[0].read_text())

        # 커스텀 어댑터의 결과가 저장됐는지 확인
        assert data["labels"][0]["custom"] is True
