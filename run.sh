#!/usr/bin/env bash
# ============================================================
# run.sh — PaddleOCR 배치 추론 실행 스크립트
#
# 사용법:
#   ./run.sh                        # 아래 설정값으로 실행
#   bash run.sh                     # 실행 권한 없을 때
# ============================================================

set -euo pipefail

# ── 경로 설정 ────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# ── 입출력 설정 ──────────────────────────────────────────────
SOURCE="/root/project/data1/vision/data/국회도서관/pdf2jpg_176"
OUTPUT="/root/project/data1/vision/paddle/paddle-pdf-batch/data/output/pdf2jpg_176"

# ── 추론 설정 ────────────────────────────────────────────────
BATCH_SIZE=32
DEVICE="gpu:0"
PRECISION="fp32"      # fp32(안정) / fp16(빠름)

# ── PDF 변환 설정 ────────────────────────────────────────────
DPI=300               # PDF→JPG 해상도 (150 / 200 / 300)
JPG_QUALITY=75        # JPEG 품질 (1~95)

# ── 입력 감지 ────────────────────────────────────────────────
if [ -d "$SOURCE" ]; then
    IMAGE_COUNT=$(find "$SOURCE" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo "[입력] 폴더 감지: $SOURCE"
    echo "[입력] 이미지 수: ${IMAGE_COUNT}장 (jpg/jpeg/png)"
elif [ -f "$SOURCE" ]; then
    echo "[입력] 파일 감지: $SOURCE"
else
    echo "[오류] 경로를 찾을 수 없습니다: $SOURCE" >&2
    exit 1
fi

echo "[출력] $OUTPUT"
echo "[설정] batch=$BATCH_SIZE  device=$DEVICE  precision=$PRECISION  dpi=$DPI  jpg_quality=$JPG_QUALITY"
echo "------------------------------------------------------------"

# ── 실행 ─────────────────────────────────────────────────────
# PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
"$VENV_PYTHON" "$SCRIPT_DIR/main.py" \
    "$SOURCE" \
    --output      "$OUTPUT" \
    --batch-size  "$BATCH_SIZE" \
    --device      "$DEVICE" \
    --precision   "$PRECISION" \
    --dpi         "$DPI" \
    --jpg-quality "$JPG_QUALITY"
