# PaddleOCR 배치 추론 파이프라인

PDF / 이미지 파일 / 폴더 / URL / 파일 경로 리스트를 입력받아  
이미지 1장당 JSON 1개를 출력하는 OCR 파이프라인.

---

## 특징

- **다양한 입력 지원**: PDF, 단일 이미지, 폴더, URL, 파일 경로 리스트
- **배치 추론**: `batch_size` 설정, 마지막 배치는 나머지 수만큼 자동 축소
- **Dry-run 모드**: `output_dir=''` 설정 시 저장 없이 결과 dict 반환
- **모델 캐시**: 동일 설정 조합은 PaddleOCR **딱 1번만 GPU에 로딩**, 이후 재사용
- **작업 큐 (Job Queue)**: 동시 요청을 순차 처리, HTTP timeout 소실 없음
- **가로 / 세로 문서 설정**: 방향 감지 플래그 3종 개별 제어
- **FastAPI 서버**: Swagger UI로 HTTP 요청 가능
- **CLI + Shell 스크립트**: `run.sh`로 간단 실행
- **어댑터 패턴**: OCR 모델 교체 시 해당 어댑터 파일만 수정

---

## 파일 구조

```
paddle-pdf-batch/
├── config.py              # 전역 설정값 (device, batch_size, pdf_dpi 등)
├── main.py                # CLI 진입점 (argparse)
├── pipeline.py            # 파이프라인 조립·실행 + run_ocr() + build_pipeline()
├── run.sh                 # Shell 실행 스크립트 (설정값만 수정해서 사용)
├── app.py                 # FastAPI 서버 + 작업 큐 (Job Queue)
├── JOB_QUEUE.md           # 작업 큐 상세 설명·예제
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

---

## 요구사항

| 항목 | 버전 |
|------|------|
| Python | 3.10 이상 |
| paddleocr | 3.4.0 |
| paddlepaddle-gpu | 3.3.0 |

```bash
pip install -r requirements_venv310.txt
```

---

## 빠른 시작

### Shell 스크립트 (권장)

`run.sh` 상단 설정값만 수정 후 실행:

```bash
# 설정 편집
vi run.sh

# 실행
./run.sh
```

실행 시 출력 예시:
```
[입력] 폴더 감지: /data/images/
[입력] 이미지 수: 176장 (jpg/jpeg/png)
[출력] /data/ocr_json/
[설정] batch=32  device=gpu:0  precision=fp32  dpi=300  jpg_quality=75
------------------------------------------------------------
대상 이미지: 176장  |  배치 크기: 32
OCR 진행: 100%|████████| 6/6 [00:12<00:00]
  [32/176]  2.54 img/s  배치 avg conf 0.923  남은시간 약 1.1분
  ...
완료: 176장  |  69.3초  |  2.54 img/s  |  전체 avg conf 0.9187
저장 경로: /data/ocr_json/
```

### CLI

```bash
cd /data1/vision/paddle/paddle-pdf-batch

# 이미지 폴더
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python main.py /data/images/ --output ./ocr_json --batch-size 8

# PDF 파일
.venv/bin/python main.py /data/문서A.pdf --output ./ocr_json --batch-size 8 --dpi 300

# 단일 이미지
.venv/bin/python main.py /data/image.jpg --output ./ocr_json

# URL
.venv/bin/python main.py https://example.com/doc.pdf --output ./ocr_json --no-ssl-verify

# 전체 옵션 확인
.venv/bin/python main.py --help
```

### Python 함수

```python
from pipeline import run_ocr

# 이미지 폴더
saved = run_ocr(
    source     = "/data/images/",
    output_dir = "/data/ocr_json/",
    batch_size = 8,
    device     = "gpu:0",
    precision  = "fp16",
)
# saved → list[Path]  저장된 JSON 경로 목록

# 파일 경로 리스트 직접 입력
saved = run_ocr(
    source     = ["/data/a.jpg", "/data/b.png", "/data/c.jpg"],
    output_dir = "/data/ocr_json/",
    batch_size = 8,
)

# Dry-run — 저장 없이 결과 dict 반환
results = run_ocr(
    source     = "/data/image.jpg",
    output_dir = "",   # 빈 문자열 → 저장 스킵
)
# results → list[dict]  result dict 목록
```

---

## 입력 형태

| 입력 | 예시 | 설명 |
|------|------|------|
| 이미지 파일 | `"/data/image.jpg"` | 단일 이미지 |
| PDF 파일 | `"/data/문서.pdf"` | 페이지별 JPG 변환 후 배치 추론 |
| 폴더 | `"/data/images/"` | 하위 이미지 전체 수집 (재귀) |
| URL | `"https://example.com/doc.pdf"` | 다운로드 후 처리 |
| 파일 리스트 | `["/a.jpg", "/b.png"]` | 경로 문자열 리스트 |

지원 이미지 확장자: `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

---

## CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `source` | (필수) | 입력 경로 또는 URL |
| `--output` | (필수) | JSON 출력 폴더 |
| `--batch-size` | `1` | 배치 크기 |
| `--device` | `gpu:0` | 추론 장치 (`gpu:1`, `cpu`) |
| `--precision` | `fp32` | `fp32`(안정) / `fp16`(빠름) |
| `--dpi` | `200` | PDF→JPG 변환 해상도 |
| `--jpg-quality` | `75` | JPEG 저장 품질 (1~95) |
| `--no-ssl-verify` | `False` | SSL 인증서 검증 비활성화 |

---

## 설정값 (config.py)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `lang` | `korean` | OCR 인식 언어 |
| `device` | `gpu:0` | 추론 장치 |
| `precision` | `fp32` | `fp16`(빠름) / `fp32`(안정) |
| `batch_size` | `1` | 기본 단일 추론 |
| `pdf_dpi` | `200` | PDF→JPG 해상도 (150 / 200 / 300) |
| `jpg_quality` | `75` | JPEG 품질 (1~95) |
| `ssl_verify` | `True` | `False` = 자체서명 인증서 허용 |
| `download_timeout` | `30` | URL 다운로드 타임아웃 (초) |
| `use_doc_orientation_classify` | `False` | 문서 회전 감지·보정 |
| `use_textline_orientation` | `False` | 텍스트라인 방향 감지 (세로쓰기) |
| `use_doc_unwarping` | `False` | 문서 왜곡 보정 (가장 느림) |

---

## 문서 방향 설정

| 상황 | `use_doc_orientation_classify` | `use_textline_orientation` | `use_doc_unwarping` |
|------|---|---|---|
| PDF 변환 이미지 (기본) | False | False | False |
| 방향 불확실한 촬영 이미지 | **True** | False | False |
| 세로쓰기 문서 (고서·한문) | False | **True** | False |
| 구겨지거나 굴곡 있는 문서 | False | False | **True** |

---

## 배치 동작

| 이미지 수 | batch_size | 실제 배치 구성 |
|---|---|---|
| 100장 | 8 | 12배치(8장) + 1배치(4장) |
| 3장 | 8 | 1배치(3장) — 자동 축소 |
| 1장 | 8 | 1배치(1장) — 자동 축소 |

---

## 출력 JSON 스키마

이미지 1장당 JSON 1개 저장:

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

- `comment` = `code` = 인식 텍스트
- `mark`: 좌상단 원점 AABB 직사각형
- `index`: 0부터 시작

### PDF 입력 시 저장 경로

```
output_dir/
└── 문서A/
    ├── 문서A_images/     ← PDF에서 변환된 JPG
    │   ├── page_001.jpg
    │   └── page_002.jpg
    └── 문서A_labels/     ← OCR 결과 JSON
        ├── page_001.json
        └── page_002.json
```

---

## 모델 캐시

`from pipeline import run_ocr` 한 줄로 모델 캐시까지 자동으로 활성화됩니다.

### 동작 원리

`pipeline.py` 모듈이 import될 때 모듈 레벨에 빈 딕셔너리가 생성됩니다:

```python
# pipeline.py — 서버/프로세스가 살아있는 동안 메모리에 상주
_model_cache: dict[tuple, object] = {}
```

`run_ocr()` 호출마다 `_get_or_create_model(config)`이 실행됩니다:

```python
key = (lang, device, precision,
       use_doc_orientation_classify,
       use_textline_orientation,
       use_doc_unwarping)

if key not in _model_cache:
    _model_cache[key] = PaddleOCR(...)   # GPU 로딩 (최초 1회, 수 초 소요)

return _model_cache[key]                 # 이후 즉시 반환
```

### 흐름

```
서버 시작
    │
    └─ from pipeline import run_ocr
           _model_cache = {}  ← 메모리에 생성

첫 번째 요청 (korean / gpu:0 / fp32)
    │
    └─ _get_or_create_model()
           key = ("korean", "gpu:0", "fp32", False, False, False)
           캐시 없음 → PaddleOCR() 초기화 → GPU 로딩 (수 초)
           _model_cache[key] = <PaddleOCR 인스턴스>

두 번째 ~ N번째 요청 (같은 설정)
    │
    └─ _get_or_create_model()
           key 동일 → 캐시 히트 → 즉시 반환  (0초, GPU 재로딩 없음)

설정이 다른 요청 (fp16으로 변경)
    │
    └─ _get_or_create_model()
           key = ("korean", "gpu:0", "fp16", False, False, False)
           캐시 없음 → PaddleOCR() 재초기화 → 새 캐시 항목 추가
           _model_cache = {
               (...fp32...): <인스턴스 A>,
               (...fp16...): <인스턴스 B>,   ← 추가됨
           }
```

### 캐시 키

모델 로딩 방식에 영향을 주는 파라미터 6가지가 캐시 키입니다:

| 키 항목 | 변경 시 |
|---|---|
| `lang` | 해당 언어 모델 재초기화 |
| `device` | GPU/CPU 재배치 |
| `precision` | fp16/fp32 재설정 |
| `use_doc_orientation_classify` | 방향 감지 모듈 로딩 여부 |
| `use_textline_orientation` | 텍스트라인 모듈 로딩 여부 |
| `use_doc_unwarping` | UVDoc 모듈 로딩 여부 |

`batch_size`, `pdf_dpi`, `jpg_quality` 등은 **런타임 파라미터**라 캐시 키에 포함되지 않습니다 — 매 호출마다 자유롭게 바꿔도 모델 재로딩 없이 동작합니다.

---

## 코드 흐름

```
run_ocr(source, output_dir, ...)
│
├─ output_dir='' → dry_run=True  → 임시 폴더 사용, save=False
└─ output_dir 지정 → dry_run=False → save=True
│
├─ _get_or_create_model(config)
│       캐시 있음 → 즉시 반환 (재로딩 없음)
│       캐시 없음 → PaddleOCR() 초기화 → 캐시 저장 → 반환
│
└─▶ Pipeline.run(source, output_dir, batch_size, save)
    │
    │ [1] InputAdapter.resolve(source, output_dir)
    │     ├─ list[str]   → 각 항목 재귀 처리 후 합산
    │     ├─ URL         → 다운로드 → 재귀 처리
    │     ├─ .pdf        → 페이지별 JPG 변환 → list[Path]
    │     ├─ 이미지 파일  → [Path]
    │     └─ 폴더        → rglob() 이미지 수집
    │              ↓
    │         list[Path]  정렬된 이미지 경로 목록
    │
    │ [2] 배치 분할  image_paths[i : i + batch_size]
    │     마지막 배치 자동 축소
    │
    │ [3] 배치 루프
    │     model.predict(batch_paths)  → list() 즉시 실체화
    │     신뢰도(rec_scores) 수집
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

---

## FastAPI 서버 + 작업 큐

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Swagger UI
http://localhost:8000/docs
```

`POST /ocr`는 `job_id`를 즉시 반환합니다. 실제 추론은 백그라운드 워커가 순차 처리합니다.

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 + 현재 큐 크기 |
| `POST` | `/ocr` | 작업 등록 → `job_id` 즉시 반환 |
| `GET` | `/status/{job_id}` | 상태 조회 (pending / running / done / error) |
| `GET` | `/result/{job_id}` | 완료 결과 조회 |
| `DELETE` | `/result/{job_id}` | 결과 메모리 삭제 |

자세한 설명 및 예제 → [`JOB_QUEUE.md`](JOB_QUEUE.md)

---

## 어댑터 교체

### OCR 모델 교체 (EasyOCR, Tesseract 등)

```python
# adapters/result_adapter.py에 새 클래스 작성 후
# pipeline.py build_pipeline() 안에서:
return Pipeline(
    model=NewOcrModel(...),
    input_adapter=InputAdapter(...),
    result_adapter=NewResultAdapter(),  # ← 여기만 교체
)
```

### 입력 소스 추가 (S3, GCS 등)

```python
# adapters/input_adapter.py에 새 클래스 작성 후
# InputAdapterProtocol을 만족하는 resolve() 구현
return Pipeline(
    model=...,
    input_adapter=NewInputAdapter(),    # ← 여기만 교체
    result_adapter=ResultAdapter(),
)
```

---

## 테스트

```bash
# 전체 (Mock 모델 사용 — GPU/PaddleOCR 불필요, 약 0.2초)
.venv/bin/python -m pytest tests/ -v

# 파일별
.venv/bin/python -m pytest tests/test_input_adapter.py -v
.venv/bin/python -m pytest tests/test_result_adapter.py -v
.venv/bin/python -m pytest tests/test_pipeline.py -v
```
