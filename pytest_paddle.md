# pytest 가이드 — paddle-pdf-batch 기준

## pytest로 알 수 있는 것

이 프로젝트의 테스트는 **실제 GPU/모델/네트워크 없이** 코드 로직만 빠르게 검증한다.
Mock(가짜 객체)으로 외부 의존성을 대체하기 때문에 0.2초 안에 47개 테스트가 완료된다.

### 현재 테스트가 검증하는 내용

**InputAdapter** (`tests/test_input_adapter.py`)

| 테스트 클래스 | 검증 내용 |
|--------------|-----------|
| `TestIsUrl` | `http://`, `https://` 판별 로직 |
| `TestScanFolder` | 확장자 필터링, 하위 폴더 재귀 수집, 정렬 재현성 |
| `TestResolveSingleImage` | 단일 파일 경로 반환, 존재하지 않는 파일 예외 |
| `TestPdfToJpg` | 파일명 형식(`page_001.jpg`), DPI 인자 전달, 폴더 구조 |
| `TestResolveUrl` | 파일명 추출, SSL 우회 경고 억제 |
| `TestResolveInvalidInput` | 미지원 확장자·존재하지 않는 경로 예외 |

**ResultAdapter** (`tests/test_result_adapter.py`)

| 테스트 클래스 | 검증 내용 |
|--------------|-----------|
| `TestGetImageSize` | PIL 헤더로 width/height 정확히 읽기 |
| `TestExtractBoxes` | `rec_boxes → rec_polys → dt_polys` 우선순위 fallback |
| `TestPolyToAabb` | 폴리곤 → AABB 좌표 수학 변환 정확성 |
| `TestMakeLabel` | UUID4 형식, index 정수 타입, mark 좌표 계산, comment=code |
| `TestConvert` | 전체 변환 흐름, 빈 결과 처리, to_dict() 없는 pred 방어 처리 |

**Pipeline** (`tests/test_pipeline.py`)

| 테스트 클래스 | 검증 내용 |
|--------------|-----------|
| `TestPipelineRun` | JSON 파일 생성, 스키마 구조, 배치 처리, 출력 폴더 자동 생성 |
| `TestDependencyInjection` | 커스텀 어댑터 교체 시 파이프라인이 해당 어댑터를 실제로 사용하는지 |

---

## pytest로 알 수 없는 것

```
pytest 전부 통과 ≠ 실제 OCR이 잘 된다
```

| 범위 밖 항목 | 이유 |
|-------------|------|
| PaddleOCR 인식 정확도 | 모델을 MockModel로 대체했기 때문 |
| GPU 메모리 사용량 / 추론 속도 | 실제 하드웨어 미사용 |
| 실제 PDF 페이지 변환 품질 | `convert_from_path`를 Mock 처리 |
| 실제 URL 다운로드 동작 | `requests.get`을 Mock 처리 |
| 대용량(2000장) 배치 안정성 | 테스트는 최대 5장 수준 |
| SSL 우회 실제 동작 | 네트워크 미연결 상태로 실행 |

---

## pytest의 실제 가치

### 로직 버그 조기 발견
코드를 잘못 작성하면 배포 전에 잡힌다.

```
예) rec_boxes가 없을 때 rec_polys로 넘어가는 fallback 코드를 빠뜨리면
    TestExtractBoxes::test_falls_back_to_rec_polys 가 즉시 실패
```

### 리팩토링 안전망
내부 구현을 바꿔도 동작이 유지되는지 확인할 수 있다.

```
예) _poly_to_aabb() 알고리즘 수정
    → TestPolyToAabb 통과하면 좌표 계산이 깨지지 않았음을 보장
```

### 어댑터 교체 검증
새 모델이나 새 어댑터로 교체할 때 기존 스키마가 깨지지 않는지 확인한다.

```
예) EasyOCR용 ResultAdapter로 교체
    → TestConvert 통과 → JSON 스키마 규격 유지 확인
```

### JSON 스키마 회귀 방지
실수로 필드 타입이나 키명을 바꾸면 테스트가 잡아준다.

```
예) index를 실수로 문자열로 반환하도록 변경
    → TestMakeLabel::test_index_is_integer 실패 → 배포 전에 발견
```

---

## 실제 OCR 품질 검증 — 스모크 테스트

pytest는 "코드 로직"을 검증하고, 스모크 테스트는 "실제 동작"을 검증한다.
두 가지를 함께 써야 "코드도 맞고, 실제도 된다"가 보장된다.

```bash
# 샘플 이미지 1장으로 실제 파이프라인 실행
python main.py /data/샘플.jpg --output ./smoke_test

# 결과 확인
cat ./smoke_test/샘플.json
```

정상 결과라면 아래 구조가 출력되어야 한다:

```json
{
  "info": { "width": 2481, "height": 3509 },
  "labels": [
    {
      "id": "f3d4a0b4-...",
      "index": 0,
      "comment": "인식된 텍스트",
      "code": "인식된 텍스트",
      "mark": { "x": 100, "y": 200, "width": 300, "height": 50, "type": "RECT" }
    }
  ]
}
```

---

## 테스트 코드 작성 기준

### 기준은 메인 스크립트가 아니라 "계약(Contract)"

메인 스크립트는 구현 방법이고, 테스트가 검증하는 건 **"이 함수는 무엇을 보장해야 하는가"**다.

```
메인 스크립트  →  "어떻게 구현했는가"
테스트 코드    →  "무엇을 보장해야 하는가"
```

구현이 바뀌어도 보장해야 할 것이 유지되면 테스트는 통과해야 한다.

---

### 기준 1 — 공개 인터페이스(Public Interface)

내부 구현이 아닌, 외부에서 호출하는 메서드를 기준으로 테스트한다.

```python
# ✅ 테스트 대상 — 외부에서 쓰는 진입점
adapter.resolve(source, jpg_dir)      # InputAdapter
adapter.convert(pred, image_path)     # ResultAdapter
pipeline.run(source, output_dir)      # Pipeline

# 내부 메서드(_로 시작)는 원칙적으로 테스트하지 않음
# 예외: 로직이 복잡하거나 독립적으로 검증할 가치가 있는 경우
#       → 이 프로젝트에서는 _is_url(), _scan_folder(), _poly_to_aabb() 등을 별도 검증
#         (내부 구현이 바뀌면 해당 테스트도 함께 수정해야 하는 트레이드오프 존재)
```

---

### 기준 2 — 스키마·계약

출력 JSON 스키마가 확정되어 있으므로, 구조가 깨지는지 검증한다.

```python
assert set(result.keys()) == {"info", "labels"}  # 최상위 키 고정
assert isinstance(label["index"], int)            # 타입 계약
assert mark["type"] == "RECT"                     # 값 계약
```

---

### 기준 3 — 경계값(Edge Case)

정상 케이스만큼 중요한 건 "비정상 입력에서도 안 터지는가"다.

```python
# 빈 결과 → 예외 없이 빈 labels 반환
pred = MockPred({"rec_texts": [], "rec_boxes": []})
assert result["labels"] == []

# to_dict() 없는 pred → 방어 처리
pred = "invalid_object"
assert result["labels"] == []

# 존재하지 않는 파일 → 적절한 예외 발생
with pytest.raises(FileNotFoundError):
    adapter.resolve("/nonexistent/image.jpg", tmp_path)
```

---

### 기준 4 — 설계 결정 고정

우선순위·규칙 같은 설계 결정은 테스트로 명시적으로 고정한다.
나중에 실수로 순서가 바뀌면 바로 잡힌다.

```python
# rec_boxes → rec_polys → dt_polys 우선순위가 설계 결정
# 각 경우를 따로 테스트해 순서가 바뀌는 실수를 방지
test_uses_rec_boxes_when_available()
test_falls_back_to_rec_polys()
test_falls_back_to_dt_polys()
test_ignores_empty_rec_boxes()   # 빈 리스트일 때도 다음으로 넘어가야 함
```

---

### 테스트 작성 흐름

```
메인 코드 작성
      │
      ▼
"이 함수가 보장해야 하는 것"을 정리
      │
      ├── 정상 케이스  : 올바른 입력 → 올바른 출력
      ├── 경계 케이스  : 빈 입력, None, 빈 리스트
      ├── 오류 케이스  : 잘못된 입력 → 적절한 예외
      └── 설계 결정   : 우선순위, 규칙, 스키마 구조
      │
      ▼
Mock으로 외부 의존성 대체
(모델, 네트워크, 파일시스템 → 테스트가 느려지거나 환경에 의존하지 않도록)
      │
      ▼
테스트 작성 → pytest 실행
```

---

## 테스트 실행 명령어

```bash
cd /data1/vision/paddle/paddle-pdf-batch
source .venv/bin/activate

# 전체 실행
pytest tests/ -v

# 파일별 실행
pytest tests/test_input_adapter.py -v
pytest tests/test_result_adapter.py -v
pytest tests/test_pipeline.py -v

# 특정 클래스만
pytest tests/test_result_adapter.py::TestConvert -v

# 특정 테스트 하나만
pytest tests/test_result_adapter.py::TestConvert::test_convert_with_rec_boxes -v
```

---

## 정리

```
pytest가 지키는 것          실제 동작은 별도 확인 필요
────────────────────        ──────────────────────────
코드 로직 정확성            OCR 인식 품질
JSON 스키마 규격            GPU 메모리 / 속도
어댑터 교체 안전성          대용량 배치 안정성
리팩토링 회귀 방지          네트워크 / 외부 서비스
```
