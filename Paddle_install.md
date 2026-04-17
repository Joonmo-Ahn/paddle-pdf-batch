# PaddleOCR 가상환경 설치 가이드

이 프로젝트에서 실제 사용 중인 환경을 기준으로 작성.

- Python 3.10
- paddlepaddle-gpu 3.3.0
- paddleocr 3.4.0
- paddlex 3.4.3
- CUDA 13.x (nvidia-cuda-runtime 13.0.88)
- 가상환경 위치: `/data1/vision/paddle/paddle-pdf-batch/.venv`

---

## 1. 사전 요건 확인

```bash
# GPU 확인
nvidia-smi

# Python 3.10 확인
python3.10 --version
# Python 3.10.x 이어야 함
```

> Python 3.10이 없으면:
> ```bash
> sudo apt install python3.10 python3.10-venv python3.10-dev
> ```

---

## 2. 가상환경 생성

```bash
cd /data1/vision/paddle/paddle-pdf-batch

python3.10 -m venv .venv
```

---

## 3. pip 업그레이드

```bash
.venv/bin/pip install --upgrade pip
```

---

## 4. PaddlePaddle GPU 설치

paddlepaddle-gpu는 CUDA 버전에 맞는 패키지를 공식 인덱스에서 받아야 한다.

```bash
# CUDA 13.x 기준 (이 환경에서 사용 중)
.venv/bin/pip install paddlepaddle-gpu==3.3.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

> **다른 CUDA 버전일 경우** 인덱스 URL의 `cu130` 부분을 변경:
>
> | CUDA 버전 | 인덱스 URL |
> |-----------|-----------|
> | CUDA 12.6 | `cu126` |
> | CUDA 12.3 | `cu123` |
> | CUDA 11.8 | `cu118` |
>
> [PaddlePaddle 공식 설치 안내](https://www.paddlepaddle.org.cn/install/quick)

설치 확인:

```bash
.venv/bin/python -c "import paddle; paddle.utils.run_check()"
# PaddlePaddle is installed successfully! ... Running with gpu 가 출력되면 정상
```

---

## 5. 나머지 패키지 설치

```bash
.venv/bin/pip install -r requirements_venv310.txt
```

> `paddlepaddle-gpu`가 이미 설치된 상태에서 실행하면  
> `requirements_venv310.txt` 안의 `paddlepaddle-gpu==3.3.0` 항목은 스킵된다.

주요 패키지 (requirements_venv310.txt 포함):

| 패키지 | 버전 |
|--------|------|
| paddlepaddle-gpu | 3.3.0 |
| paddleocr | 3.4.0 |
| paddlex | 3.4.3 |
| fastapi | 0.135.3 |
| uvicorn | 0.44.0 |
| PyMuPDF | 1.27.2.2 |
| pdf2image | 1.17.0 |
| opencv-python | 4.13.0.92 |
| numpy | 2.2.6 |
| pillow | 12.2.0 |

---

## 6. 설치 확인

```bash
# 패키지 확인
.venv/bin/pip show paddleocr paddlex paddlepaddle-gpu

# PaddleOCR import 확인
.venv/bin/python -c "
import paddle
import paddleocr
import paddlex
print('paddle:', paddle.__version__)
print('paddleocr:', paddleocr.__version__)
print('paddlex:', paddlex.__version__)
print('GPU 수:', paddle.device.cuda.device_count())
"
```

정상 출력 예시:

```
paddle: 3.3.0
paddleocr: 3.4.0
paddlex: 3.4.3
GPU 수: 1
```

---

## 7. 환경 변수 설정

PaddlePaddle은 import 시 모델 소스를 확인하는 네트워크 요청을 보낸다.  
아래 환경 변수를 설정하면 이 체크를 건너뛰어 import 속도가 빨라진다.

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

매번 export 없이 사용하려면 `.bashrc` 또는 `.zshrc`에 추가:

```bash
echo 'export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True' >> ~/.bashrc
source ~/.bashrc
```

---

## 8. 첫 실행 (모델 자동 다운로드)

최초 실행 시 PaddleOCR 모델이 자동 다운로드된다 (수백 MB, 인터넷 연결 필요).

```bash
cd /data1/vision/paddle/paddle-pdf-batch

PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python -c "
from pipeline import run_ocr
print('모델 로딩 중...')
result = run_ocr(source='<이미지경로>', output_dir='./test_out')
print('완료:', result)
"
```

모델 캐시 기본 위치: `~/.paddleocr/`

---

## 9. FastAPI 서버 실행

```bash
cd /data1/vision/paddle/paddle-pdf-batch

PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

브라우저에서 Swagger UI 확인: `http://localhost:8000/docs`

---

## 10. CLI 실행

```bash
cd /data1/vision/paddle/paddle-pdf-batch

# 이미지 폴더
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python main.py /data/images/ --output ./results --batch-size 8

# PDF 파일
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
.venv/bin/python main.py /data/문서.pdf --output ./results --dpi 300

# Shell 스크립트 (run.sh 상단 변수 수정 후)
./run.sh
```

---

## 트러블슈팅

### `ImportError: libcudart.so` 또는 CUDA 관련 오류

CUDA 드라이버와 paddlepaddle-gpu 버전이 맞지 않는 경우.  
`nvidia-smi`로 드라이버 버전을 확인하고 맞는 `cu***` 인덱스로 재설치.

```bash
nvidia-smi | head -5
# CUDA Version: 13.0 이면 cu130 사용
```

### `ModuleNotFoundError: No module named 'paddle'`

가상환경 활성화 없이 실행한 경우.  
`.venv/bin/python`을 명시적으로 사용하거나 `source .venv/bin/activate` 후 실행.

### 모델 다운로드 실패

방화벽으로 외부 접근이 막힌 환경이면 모델을 수동으로 옮겨야 한다.  
모델 경로: `~/.paddleocr/` (기본값)

### `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` 미설정 시

import 시 외부 네트워크 요청 발생 → 방화벽 환경에서 수 초~수십 초 지연.  
항상 환경 변수를 설정하고 실행할 것.
