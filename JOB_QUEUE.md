# 작업 큐 (Job Queue)

동시에 여러 요청이 들어와도 GPU를 순차적으로 사용하고,  
HTTP timeout 없이 안전하게 결과를 받을 수 있는 비동기 작업 처리 방식.

---

## 왜 필요한가

### 문제: 동기 방식의 한계

```
클라이언트 A (1000장) ──▶ GPU 추론 중 (10분)
클라이언트 B (2000장) ──▶ 대기 중... (20분 걸릴 예정)
클라이언트 C (500장)  ──▶ 대기 중... (30분 걸릴 예정)
                               ↑ HTTP 연결을 붙잡고 기다림
                               ↑ 클라이언트/네트워크 timeout → 소실
```

HTTP 연결이 GPU 작업이 끝날 때까지 살아있어야 합니다.  
중간에 연결이 끊기면 결과를 받을 수 없습니다.

### 해결: 큐 방식

```
클라이언트 A ──▶ POST /ocr ──▶ {"job_id": "aaa"} 즉시 반환
클라이언트 B ──▶ POST /ocr ──▶ {"job_id": "bbb"} 즉시 반환  ← 연결 끊김
클라이언트 C ──▶ POST /ocr ──▶ {"job_id": "ccc"} 즉시 반환  ← 연결 끊김

                백그라운드에서 순차 처리
                A 완료 → B 시작 → C 시작

클라이언트 A ──▶ GET /result/aaa ──▶ 결과 조회 (언제든 가능)
```

HTTP 연결은 `job_id`를 받는 순간 끊어집니다.  
결과는 나중에 언제든 조회할 수 있습니다.

---

## 동작 원리

```
┌─────────────────────────────────────────────────────────────┐
│  HTTP 스레드 (FastAPI)          백그라운드 워커 스레드        │
│                                                             │
│  POST /ocr                                                  │
│    │                                                        │
│    ├─ job_id 생성                                           │
│    ├─ 큐에 put() ──────────────▶ get() 꺼냄                 │
│    └─ job_id 즉시 반환          run_ocr() 실행              │
│         (연결 끊김)              결과 저장 (dict)            │
│                                                             │
│  GET /status/{job_id}           A 완료 → B get() → ...     │
│    └─ dict 조회 → 상태 반환                                  │
│                                                             │
│  GET /result/{job_id}                                       │
│    └─ dict 조회 → 결과 반환                                  │
└─────────────────────────────────────────────────────────────┘
```

**핵심 컴포넌트:**

| 컴포넌트 | 역할 |
|---|---|
| `queue.Queue` | thread-safe FIFO 큐. HTTP 스레드가 `put()`, 워커가 `get()` |
| 워커 스레드 1개 | GPU를 순차적으로 점유. `daemon=True`라 서버 종료 시 자동 종료 |
| `_job_results dict` | job_id → 상태/결과 저장소. `threading.Lock`으로 보호 |

---

## 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/health` | 서버 상태 + 현재 큐 크기 |
| `POST` | `/ocr` | 작업 등록 → `job_id` 즉시 반환 |
| `GET` | `/status/{job_id}` | 작업 상태 조회 |
| `GET` | `/result/{job_id}` | 완료된 결과 조회 |
| `DELETE` | `/result/{job_id}` | 결과 메모리에서 삭제 |

### 작업 상태 흐름

```
등록     대기      추론 중    완료
pending → pending → running → done
                           → error (실패 시)
```

---

## 빠르게 시작하기

### 서버 실행

```bash
cd /data1/vision/paddle/paddle-pdf-batch

PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### Swagger UI

```
http://localhost:8000/docs
```

---

## 사용 예제

### curl

```bash
# 1. 작업 등록
curl -X POST http://localhost:8000/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "source":     "/data/images/",
    "output_dir": "/data/ocr_json/",
    "batch_size": 8,
    "device":     "gpu:0",
    "precision":  "fp16"
  }'

# 응답 (즉시)
# {"job_id": "f3d4a0b4-...", "status": "pending", "queue_position": 0}

# 2. 상태 조회 (폴링)
curl http://localhost:8000/status/f3d4a0b4-...
# {"job_id": "f3d4a0b4-...", "status": "running", "queue_position": null}

# 3. 결과 조회 (done 상태일 때)
curl http://localhost:8000/result/f3d4a0b4-...
# {
#   "job_id": "f3d4a0b4-...",
#   "status": "done",
#   "saved_count": 176,
#   "saved_paths": ["/data/ocr_json/000001.json", ...],
#   "output_dir": "/data/ocr_json/"
# }

# 4. 서버 상태 + 큐 크기 확인
curl http://localhost:8000/health
# {"status": "ok", "queue_size": 2}
```

### Python

```python
import time
import requests

BASE = "http://localhost:8000"

# 1. 작업 등록
res = requests.post(f"{BASE}/ocr", json={
    "source":     "/data/images/",
    "output_dir": "/data/ocr_json/",
    "batch_size": 8,
    "device":     "gpu:0",
})
job_id = res.json()["job_id"]
print(f"등록 완료: {job_id}")

# 2. 완료까지 폴링
while True:
    status = requests.get(f"{BASE}/status/{job_id}").json()
    print(f"상태: {status['status']}")

    if status["status"] == "done":
        break
    if status["status"] == "error":
        print("오류 발생")
        break

    time.sleep(10)   # 10초마다 확인

# 3. 결과 조회
result = requests.get(f"{BASE}/result/{job_id}").json()
print(f"완료: {result['saved_count']}장")
print(f"저장 경로: {result['output_dir']}")
```

### 여러 작업 동시 등록

```python
import requests

BASE = "http://localhost:8000"

datasets = [
    {"source": "/data/set_a/", "output_dir": "/data/out_a/", "batch_size": 8},
    {"source": "/data/set_b/", "output_dir": "/data/out_b/", "batch_size": 8},
    {"source": "/data/set_c/", "output_dir": "/data/out_c/", "batch_size": 8},
]

# 모두 즉시 등록 (큐에 쌓임)
job_ids = []
for d in datasets:
    res = requests.post(f"{BASE}/ocr", json=d)
    job_ids.append(res.json()["job_id"])
    print(f"등록: {res.json()['job_id']}  대기 순서: {res.json()['queue_position']}")

# 등록: aaa...  대기 순서: 0   ← 바로 실행
# 등록: bbb...  대기 순서: 1   ← aaa 끝나면 실행
# 등록: ccc...  대기 순서: 2   ← bbb 끝나면 실행
```

---

## 상태별 응답 예시

### pending (큐 대기 중)
```json
{
  "job_id": "f3d4a0b4-...",
  "status": "pending",
  "queue_position": 2
}
```

### running (추론 중)
```json
{
  "job_id": "f3d4a0b4-...",
  "status": "running",
  "queue_position": null
}
```

### done (완료)
```json
{
  "job_id": "f3d4a0b4-...",
  "status": "done",
  "saved_count": 176,
  "saved_paths": ["/data/ocr_json/000001.json", "..."],
  "output_dir": "/data/ocr_json/"
}
```

### error (실패)
```json
{
  "job_id": "f3d4a0b4-...",
  "status": "error",
  "detail": "파일을 찾을 수 없습니다: /data/wrong_path"
}
```

---

## 주의사항

### 결과 메모리 관리

결과는 서버 메모리에 보관됩니다. 장시간 운영 시 아래 방법으로 정리하세요:

```bash
# 완료된 작업 결과 삭제
curl -X DELETE http://localhost:8000/result/f3d4a0b4-...
```

서버를 재시작하면 메모리의 모든 결과가 초기화됩니다.  
(단, 이미 저장된 JSON 파일은 디스크에 남아 있습니다.)

### 서버 재시작 시 큐 초기화

현재 구현은 인메모리 큐입니다. 서버 재시작 시 대기 중이던 작업은 소실됩니다.  
영구 보관이 필요하면 **Celery + Redis** 방식으로 전환하세요.

### 변경된 파일

큐 기능은 **`app.py`만 변경**되었습니다:

| 파일 | 변경 여부 |
|------|-----------|
| `app.py` | ✅ 변경 (큐 추가) |
| `pipeline.py` | 변경 없음 |
| `adapters/` | 변경 없음 |
| `config.py` | 변경 없음 |
| `main.py` | 변경 없음 |
| `run.sh` | 변경 없음 |
