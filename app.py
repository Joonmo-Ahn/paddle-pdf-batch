"""
app.py — FastAPI 서버

역할:
    run_ocr()을 HTTP 엔드포인트로 노출한다.
    작업 큐(Job Queue)를 통해 동시 요청을 순차적으로 처리한다.
    Swagger UI(http://host:8000/docs)에서 파라미터를 직접 입력해 실행 가능.

실행:
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \\
    .venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000

설계 원칙:
    - app.py는 HTTP 레이어 + 큐 관리만 담당. OCR 로직은 pipeline.run_ocr()에 위임.
    - POST /ocr 는 job_id를 즉시 반환 → HTTP timeout 소실 없음.
    - 백그라운드 워커 스레드 1개가 큐에서 순차적으로 꺼내 GPU 추론.
    - 모델은 첫 추론 시 초기화되고 캐시에 보관 → 이후 재로딩 없음.
"""

import queue
import threading
import traceback
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from pipeline import run_ocr


# ── FastAPI 앱 생성 ───────────────────────────────────────────────────────────

app = FastAPI(
    title="PaddleOCR 배치 추론 API",
    description=(
        "PDF / 이미지 파일 / 폴더 / URL을 입력받아 "
        "이미지 1장당 JSON 1개를 생성하는 OCR 파이프라인 API.\n\n"
        "**작업 큐 방식**: `POST /ocr`는 `job_id`를 즉시 반환합니다. "
        "`GET /status/{job_id}`로 진행 상태를, "
        "`GET /result/{job_id}`로 완료된 결과를 조회하세요.\n\n"
        "출력 JSON 스키마: `{info: {width, height}, labels: [{id, index, comment, code, mark}]}`"
    ),
    version="2.0.0",
)


# ── 작업 큐 + 결과 저장소 ─────────────────────────────────────────────────────

# FIFO 큐: HTTP 스레드가 put(), 워커 스레드가 get()
# queue.Queue는 thread-safe — 별도 Lock 없이 안전하게 사용 가능
_job_queue: queue.Queue = queue.Queue()

# job_id → 상태/결과 dict. 서버 재시작 전까지 메모리에 보관.
# 읽기/쓰기가 여러 스레드에서 발생하므로 Lock으로 보호.
_job_results: dict = {}
_results_lock = threading.Lock()


def _worker() -> None:
    """
    백그라운드 워커 스레드.

    큐에서 작업을 하나씩 꺼내 순차적으로 run_ocr()을 실행한다.
    스레드 1개 = GPU 동시 점유 1개 → CUDA OOM 방지.
    daemon=True 이므로 메인 프로세스 종료 시 자동 종료.
    """
    while True:
        job_id, req = _job_queue.get()

        # 상태 → running
        with _results_lock:
            _job_results[job_id]["status"] = "running"

        try:
            saved: list[Path] = run_ocr(
                source       = req.source,
                output_dir   = req.output_dir,
                batch_size        = req.batch_size,
                pdf_dpi           = req.pdf_dpi,
                jpg_quality       = req.jpg_quality,
                ssl_verify        = req.ssl_verify,
                download_timeout  = req.download_timeout,
                lang      = req.lang,
                device    = req.device,
                precision = req.precision,
                use_doc_orientation_classify = req.use_doc_orientation_classify,
                use_textline_orientation     = req.use_textline_orientation,
                use_doc_unwarping            = req.use_doc_unwarping,
            )
            with _results_lock:
                _job_results[job_id].update({
                    "status":      "done",
                    "saved_count": len(saved),
                    "saved_paths": [str(p) for p in saved],
                    "output_dir":  req.output_dir,
                })

        except FileNotFoundError as e:
            with _results_lock:
                _job_results[job_id].update({"status": "error", "detail": str(e)})
        except ValueError as e:
            with _results_lock:
                _job_results[job_id].update({"status": "error", "detail": str(e)})
        except Exception as e:
            traceback.print_exc()
            with _results_lock:
                _job_results[job_id].update({"status": "error", "detail": str(e)})
        finally:
            _job_queue.task_done()


# 서버 시작과 동시에 워커 스레드 1개 기동
threading.Thread(target=_worker, daemon=True).start()


# ── 요청 스키마 ───────────────────────────────────────────────────────────────

class OcrRequest(BaseModel):
    """
    POST /ocr 요청 바디.
    Swagger UI에서 각 필드에 설명과 기본값이 표시된다.
    """

    # ── 필수 파라미터 ────────────────────────────────────────────────────────
    source: str = Field(
        ...,
        description=(
            "입력 경로 또는 URL.\n"
            "- 이미지 파일: `/data/image.jpg`\n"
            "- PDF 파일:   `/data/문서.pdf`\n"
            "- 폴더:       `/data/images/`\n"
            "- URL:        `https://example.com/doc.pdf`"
        ),
    )
    output_dir: str = Field(
        ...,
        description="JSON 결과물 및 중간 JPG 파일을 저장할 폴더 경로.",
        examples=["./results"],
    )

    # ── 런타임 파라미터 ───────────────────────────────────────────────────────
    batch_size: int = Field(default=1, ge=1, le=256,
        description="한 번에 GPU에 올릴 이미지 수. 클수록 처리량↑ VRAM↑.")
    pdf_dpi: int = Field(default=200, ge=72, le=600,
        description="PDF → JPG 변환 해상도. 200 DPI 권장.")
    jpg_quality: int = Field(default=75, ge=1, le=95,
        description="JPEG 저장 품질. 75가 속도·용량·품질의 균형점.")
    ssl_verify: bool = Field(default=True,
        description="False로 설정하면 자체서명 SSL 인증서도 허용.")
    download_timeout: int = Field(default=30, ge=1,
        description="URL 다운로드 최대 대기 시간(초).")

    # ── 모델 파라미터 ─────────────────────────────────────────────────────────
    lang: str = Field(default="korean",
        description="OCR 인식 언어. 변경 시 최초 1회 초기화.")
    device: str = Field(default="gpu:0",
        description="추론 장치. 예: `gpu:0`, `gpu:1`, `cpu`.")
    precision: str = Field(default="fp32",
        description="`fp32`(안정) 또는 `fp16`(빠름, VRAM 절약).")

    # ── 문서 방향 설정 ────────────────────────────────────────────────────────
    use_doc_orientation_classify: bool = Field(default=False,
        description="문서 회전 감지·보정 (0/90/180/270°). 스캔본·사진 등에 사용.")
    use_textline_orientation: bool = Field(default=False,
        description="텍스트라인 방향 감지 (가로↔세로). 세로쓰기 문서에 사용.")
    use_doc_unwarping: bool = Field(default=False,
        description="문서 왜곡(구김·굴곡) 보정. 처리 단계 중 가장 느림.")


# ── 응답 스키마 ───────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    """POST /ocr 응답 — 즉시 반환."""
    job_id:         str
    status:         str   # "pending"
    queue_position: int   # 현재 대기 순서 (0 = 바로 실행 예정)


class StatusResponse(BaseModel):
    """GET /status/{job_id} 응답."""
    job_id:         str
    status:         str              # pending / running / done / error
    queue_position: Optional[int]    # pending 상태일 때만 존재


class ResultResponse(BaseModel):
    """GET /result/{job_id} 응답 — done 상태에서만 반환."""
    job_id:      str
    status:      str
    saved_count: int
    saved_paths: list[str]
    output_dir:  str


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", summary="서버 상태 확인")
def health():
    """서버 정상 동작 여부 + 현재 큐 상태를 반환."""
    return {
        "status":     "ok",
        "queue_size": _job_queue.qsize(),
    }


@app.post(
    "/ocr",
    response_model=JobResponse,
    summary="OCR 작업 등록",
    description=(
        "OCR 작업을 큐에 등록하고 **즉시** `job_id`를 반환합니다.\n\n"
        "실제 추론은 백그라운드 워커가 순차적으로 처리합니다.\n"
        "`GET /status/{job_id}`로 진행 상태를 폴링하세요."
    ),
)
def ocr(req: OcrRequest) -> JobResponse:
    """작업을 큐에 등록하고 job_id를 즉시 반환."""
    job_id = str(uuid4())
    position = _job_queue.qsize()

    with _results_lock:
        _job_results[job_id] = {
            "status":         "pending",
            "queue_position": position,
        }

    _job_queue.put((job_id, req))

    return JobResponse(job_id=job_id, status="pending", queue_position=position)


@app.get(
    "/status/{job_id}",
    response_model=StatusResponse,
    summary="작업 상태 조회",
)
def status(job_id: str) -> StatusResponse:
    """
    작업 상태를 반환한다.

    - `pending` : 큐 대기 중
    - `running` : GPU 추론 중
    - `done`    : 완료 (`GET /result/{job_id}`로 결과 조회)
    - `error`   : 실패 (detail 필드에 오류 메시지)
    """
    with _results_lock:
        job = _job_results.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' 를 찾을 수 없습니다.")

    return StatusResponse(
        job_id         = job_id,
        status         = job["status"],
        queue_position = job.get("queue_position"),
    )


@app.get(
    "/result/{job_id}",
    response_model=ResultResponse,
    summary="작업 결과 조회",
)
def result(job_id: str) -> ResultResponse:
    """
    완료된 작업의 결과를 반환한다.
    `status`가 `done`이 아니면 404를 반환한다.
    """
    with _results_lock:
        job = _job_results.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' 를 찾을 수 없습니다.")

    if job["status"] != "done":
        raise HTTPException(
            status_code=404,
            detail=f"아직 완료되지 않았습니다. 현재 상태: {job['status']}",
        )

    return ResultResponse(
        job_id      = job_id,
        status      = job["status"],
        saved_count = job["saved_count"],
        saved_paths = job["saved_paths"],
        output_dir  = job["output_dir"],
    )


@app.delete(
    "/result/{job_id}",
    summary="작업 결과 삭제",
)
def delete_result(job_id: str):
    """메모리에서 작업 결과를 삭제한다. 장시간 운영 시 메모리 관리에 사용."""
    with _results_lock:
        if job_id not in _job_results:
            raise HTTPException(status_code=404, detail=f"job_id '{job_id}' 를 찾을 수 없습니다.")
        del _job_results[job_id]

    return {"deleted": job_id}
