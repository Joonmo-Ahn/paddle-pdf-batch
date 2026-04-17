# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

PaddleOCR 기반 배치 OCR 파이프라인.
PDF / 이미지 파일 / 폴더 / URL / 파일 경로 리스트를 입력받아 이미지 1장당 JSON 1개를 출력한다.

**설계 원칙**
- 어댑터 패턴: 모델 교체·입출력 포맷 변경 시 해당 어댑터 파일만 수정
- 의존성 주입(DI): Pipeline이 어댑터를 외부에서 주입받아 테스트·교체가 용이
- 단일 책임: 각 파일이 하나의 역할만 담당
- FastAPI + Gunicorn 서버로의 확장을 염두에 두고 설계

## 파일 구조

```
paddle-pdf-batch/
├── config.py              # 전역 설정값 (device, batch_size, pdf_dpi 등)
├── main.py                # CLI 진입점 (argparse)
├── pipeline.py            # 파이프라인 조립·실행 + run_ocr() + build_pipeline()
├── run.sh                 # Shell 실행 스크립트 (상단 변수만 수정해서 사용)
├── app.py                 # FastAPI 서버 + 작업 큐 (Job Queue)
├── JOB_QUEUE.md           # 작업 큐 설명·예제 문서
├── adapters/
│   ├── input_adapter.py   # 입력 정규화: PDF/이미지/폴더/URL/리스트 → Path 목록
│   └── result_adapter.py  # 결과 변환: PaddleOCR raw → 내부 JSON 스키마
├── utils/
│   └── id_gen.py          # UUID4 생성 (label id 용)
├── tests/
│   ├── test_input_adapter.py
│   ├── test_result_adapter.py
│   └── test_pipeline.py
└── pytest_paddle.md       # pytest 범위·한계·스모크 테스트 가이드
```

## 가상환경

```
OCR 추론 + 테스트 (Python 3.10 — PaddleOCR 설치됨, root 소유)
    /data1/vision/paddle/paddle-pdf-batch/.venv
    paddle 3.3.0 / paddleocr 3.4.0 / paddlex 3.4.3 / GPU 1개 확인
```

```bash
# Shell 스크립트 실행 (권장)
./run.sh

# CLI 실행
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python main.py <source> --output <dir> --batch-size 8

# 테스트 실행
.venv/bin/python -m pytest tests/ -v

# FastAPI 서버 실행
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

> `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` — 모델 소스 체크(외부 네트워크 요청)를 건너뛰어 첫 import 속도를 높임.

## 실행 방법

```bash
# Shell 스크립트 (run.sh 상단 변수 수정 후 실행)
./run.sh

# 이미지 폴더
python main.py /data/images/ --output ./results --batch-size 8

# PDF 파일
python main.py /data/문서A.pdf --output ./results --batch-size 8 --dpi 300

# 단일 이미지
python main.py /data/image.jpg --output ./results

# URL
python main.py https://example.com/doc.pdf --output ./results --no-ssl-verify

# CPU 추론
python main.py /data/image.jpg --output ./results --device cpu --precision fp32

# 전체 옵션
python main.py --help
```

## 입력 형태

| 입력 | 처리 방식 |
|------|----------|
| `str` — 이미지 파일 | `[Path]` 그대로 반환 |
| `str` — PDF 파일 | 페이지별 JPG 변환 → list[Path] |
| `str` — 폴더 | `rglob()` 이미지 수집 → list[Path] |
| `str` — URL | 다운로드 후 재귀 처리 |
| `list[str]` | 각 항목을 재귀 처리 후 합산 |

지원 확장자: `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

## 배치 동작

- `image_paths[i : i + batch_size]` 슬라이싱으로 자동 처리
- 마지막 배치가 `batch_size`보다 작아도 자동 축소 (별도 처리 불필요)
- `model.predict()` 반환값을 `list()`로 즉시 실체화 — 제너레이터 두 번 소비 방지

## run_ocr() 함수

`pipeline.py`의 단일 진입점. FastAPI / 외부 스크립트에서 호출.

```python
from pipeline import run_ocr   # 이 시점에 _model_cache = {} 가 메모리에 생성됨

# 저장 모드
saved = run_ocr(
    source     = "/data/images/",     # str 또는 list[str]
    output_dir = "/data/ocr_json/",
    batch_size = 8,
    device     = "gpu:0",
    precision  = "fp16",
)
# 반환: list[Path] — 저장된 JSON 경로 목록

# Dry-run 모드 (output_dir='' → 저장 없음)
results = run_ocr(
    source     = "/data/image.jpg",
    output_dir = "",                  # 빈 문자열
)
# 반환: list[dict] — result dict 목록
# 내부적으로 tempfile 사용 → 종료 시 자동 삭제
```

**파라미터 두 종류:**
- 런타임 파라미터 (`batch_size`, `pdf_dpi`, `jpg_quality` 등): 매 호출마다 자유롭게 변경. 모델 재로딩 없음.
- 모델 파라미터 (`lang`, `device`, `precision`, `use_*` 플래그): 변경 시 최초 1회 재초기화, 이후 캐시 재사용.

## 모델 캐시 (_model_cache)

`pipeline.py` 모듈 레벨에 선언된 딕셔너리. `import pipeline` 시 1번 생성되어 프로세스가 살아있는 동안 메모리에 유지된다.

```python
# pipeline.py 모듈 레벨 (import 시 딱 1번 실행)
_model_cache: dict[tuple, object] = {}
```

**캐시 키**: 모델 초기화 방식에 영향을 주는 파라미터 6개의 튜플

```python
key = (lang, device, precision,
       use_doc_orientation_classify,
       use_textline_orientation,
       use_doc_unwarping)
```

**동작 흐름:**

```
run_ocr() 호출
    │
    └─ _get_or_create_model(config)
           │
           ├─ key in _model_cache?
           │       YES → 캐시된 인스턴스 즉시 반환  (재로딩 없음)
           │       NO  → PaddleOCR(...)  GPU 로딩 (수 초)
           │              _model_cache[key] = 인스턴스
           │              반환
           │
           └─ 반환된 인스턴스로 predict() 실행
```

**캐시가 쌓이는 예시:**

```python
run_ocr(..., precision="fp32")  # 최초 → GPU 로딩
run_ocr(..., precision="fp32")  # 캐시 히트 → 즉시
run_ocr(..., precision="fp16")  # 새 조합 → 재로딩
run_ocr(..., precision="fp16")  # 캐시 히트 → 즉시

# _model_cache = {
#     (..., "fp32", ...): <PaddleOCR A>,
#     (..., "fp16", ...): <PaddleOCR B>,
# }
```

**`batch_size`, `pdf_dpi`, `jpg_quality`는 캐시 키가 아님** — 모델 초기화와 무관한 런타임 파라미터이므로 매 호출마다 자유롭게 변경 가능.

## Pipeline.run() 파라미터

```python
pipeline.run(
    source,
    output_dir,
    batch_size = 1,
    save       = True,   # False → JSON 저장 스킵, list[dict] 반환
)
```

## 아키텍처 흐름

```
run_ocr(source, output_dir, ...)
│
├─ output_dir='' → dry_run=True  → tempfile 임시 폴더, save=False
└─ output_dir 지정 → dry_run=False → save=True
│
├─ _get_or_create_model(config)   ← 캐시 히트 시 재로딩 없음
│
└─▶ Pipeline.run(source, output_dir, batch_size, save)
    │
    │ [1] InputAdapter.resolve(source, output_dir)
    │     ├─ list[str]   → 각 항목 재귀 처리 후 합산
    │     ├─ URL         → 다운로드 → 재귀 처리
    │     ├─ .pdf        → 페이지별 JPG 변환 → list[Path]
    │     ├─ 이미지 파일  → [Path]
    │     └─ 폴더        → rglob() 이미지 수집
    │              ↓ list[Path] (정렬된 이미지 경로 목록)
    │
    │ [2] 배치 분할  image_paths[i : i + batch_size]
    │     마지막 배치 자동 축소
    │
    │ [3] 배치 루프
    │     model.predict(batch_paths) → list() 즉시 실체화
    │     rec_scores / rec_score 키로 신뢰도 수집
    │     ResultAdapter.convert(pred, img_path)
    │       → { info: {width, height}, labels: [...] }
    │     배치 완료마다: img/s + 배치 avg conf + 남은시간 출력
    │
    │ [4] 저장 분기
    │     save=True  → _save_json()  → list[Path] 누적
    │     save=False → result dict   → list[dict] 누적
    │
    └─▶ 완료 통계 출력 (img/s, 전체 avg conf) → 반환
```

## 테스트

```bash
# 전체 (Mock 모델 사용 — GPU/PaddleOCR 불필요, 약 0.2초)
.venv/bin/python -m pytest tests/ -v

# 파일별
.venv/bin/python -m pytest tests/test_input_adapter.py -v
.venv/bin/python -m pytest tests/test_result_adapter.py -v
.venv/bin/python -m pytest tests/test_pipeline.py -v

# 클래스·단일 테스트
.venv/bin/python -m pytest tests/test_result_adapter.py::TestConvert -v
```

pytest의 범위·한계·스모크 테스트 방법 → `pytest_paddle.md` 참고

## 출력 JSON 스키마

```json
{
  "info": { "width": 2481, "height": 3509 },
  "labels": [
    {
      "id": "f3d4a0b4-725f-4d9d-961e-522c390072f9",
      "index": 0,
      "comment": "인식된 텍스트",
      "code": "인식된 텍스트",
      "mark": { "x": 861, "y": 393, "width": 94, "height": 130, "type": "RECT" }
    }
  ]
}
```

- `comment` = `code` = 인식 텍스트 (현재 동일값)
- `mark` 좌표계: 좌상단 원점, AABB 직사각형
- `index`: 0부터 시작하는 정수

## 설정값 (config.py)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `lang` | `korean` | OCR 인식 언어 |
| `device` | `gpu:0` | 추론 장치 (`gpu:1`, `cpu` 가능) |
| `precision` | `fp32` | `fp16`(빠름) / `fp32`(안정) |
| `batch_size` | `1` | 기본 단일 추론 |
| `pdf_dpi` | `200` | PDF→JPG 변환 해상도 (150~300, 200 권장) |
| `jpg_quality` | `75` | JPEG 저장 품질 (1~95) |
| `ssl_verify` | `True` | `False` = 자체서명 인증서 허용 |
| `download_timeout` | `30` | URL 다운로드 타임아웃(초) |
| `use_doc_orientation_classify` | `False` | 문서 회전 감지·보정 |
| `use_textline_orientation` | `False` | 텍스트라인 방향 감지 (세로쓰기) |
| `use_doc_unwarping` | `False` | 문서 왜곡 보정 (가장 느림) |

## 문서 방향 설정 가이드

| 상황 | `use_doc_orientation_classify` | `use_textline_orientation` | `use_doc_unwarping` |
|------|---|---|---|
| PDF 변환 이미지 (기본) | False | False | False |
| 방향 불확실한 촬영 이미지 | True | False | False |
| 세로쓰기 문서 (고서·한문) | False | True | False |
| 구겨지거나 굴곡 있는 문서 | False | False | True |

## FastAPI 서버 + 작업 큐 (app.py)

`pipeline.run_ocr()`을 HTTP로 노출. **작업 큐(Job Queue)** 방식으로 동시 요청을 순차 처리한다.

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Swagger UI
http://localhost:8000/docs
```

**엔드포인트:**

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 + 현재 큐 크기 |
| `POST` | `/ocr` | 작업 등록 → `job_id` 즉시 반환 |
| `GET` | `/status/{job_id}` | 상태 조회 (pending / running / done / error) |
| `GET` | `/result/{job_id}` | 완료 결과 조회 |
| `DELETE` | `/result/{job_id}` | 결과 메모리 삭제 |

**작업 큐 동작 원리:**
- `POST /ocr` → `job_id` 즉시 반환 (HTTP 연결 즉시 종료 → timeout 소실 없음)
- 백그라운드 워커 스레드 1개가 `queue.Queue`에서 순차적으로 꺼내 GPU 추론
- 결과는 `_job_results` dict(인메모리)에 보관 → `GET /result/{job_id}`로 조회
- 서버 재시작 시 미완료 작업·결과 초기화 (디스크 저장 파일은 유지)

**모델 캐시 키**: `(lang, device, precision, use_doc_orientation_classify, use_textline_orientation, use_doc_unwarping)` — 같은 조합은 재로딩 없이 재사용. 자세한 내용 → 위 `모델 캐시` 섹션 참고.

자세한 큐 설명·예제 → `JOB_QUEUE.md`

## PDF→JPG 저장 경로

```
입력:  /data/문서A.pdf
출력:
    output_dir/문서A/문서A_images/page_001.jpg   ← 변환 이미지
    output_dir/문서A/문서A_labels/page_001.json  ← OCR 결과
```

URL 다운로드 시 원본 저장 위치: `output_dir/_downloads/파일명`

## PaddleOCR bbox 버전 대응

`ResultAdapter._extract_boxes()`가 버전별 키 차이를 자동 처리:

| 우선순위 | 키 | 형태 |
|---------|-----|------|
| 1 | `rec_boxes` | AABB `[x1, y1, x2, y2]` 직접 사용 |
| 2 | `rec_polys` | 폴리곤 → numpy min/max → AABB 변환 |
| 3 | `dt_polys` | 탐지 전용 폴리곤 → AABB 변환 (fallback) |

## 어댑터 교체 방법

### OCR 모델 교체 (EasyOCR, Tesseract 등)

1. `adapters/result_adapter.py`에 새 클래스 작성 (`ResultAdapterProtocol` 만족)
2. `pipeline.py`의 `build_pipeline()`에서 교체

```python
return Pipeline(
    model=NewOcrModel(...),
    input_adapter=InputAdapter(...),
    result_adapter=NewResultAdapter(),  # ← 여기만 교체
)
```

### 입력 소스 추가 (S3, GCS 등)

1. `adapters/input_adapter.py`에 새 클래스 작성 (`InputAdapterProtocol` 만족)
2. `build_pipeline()`에서 `input_adapter=NewInputAdapter()` 로 교체
