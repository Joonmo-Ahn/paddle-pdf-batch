"""
pipeline.py — OCR 파이프라인 조립 및 실행

역할:
    InputAdapter, OCR 모델, ResultAdapter를 조합해
    '입력 → OCR 추론 → JSON 저장'의 전체 흐름을 실행한다.

의존성 주입(DI) 패턴:
    Pipeline 클래스가 어댑터를 직접 생성하지 않고 외부에서 주입받는다.
    장점:
    - 어댑터 교체 시 Pipeline 코드를 수정하지 않아도 됨
    - 테스트 시 Mock 어댑터를 주입해 실제 모델 없이 테스트 가능

FastAPI 확장 대비:
    run()이 순수한 처리 함수로 설계되어 있어
    추후 api.py에서 pipeline.run()을 그대로 호출하면 됨.
    (main.py와 api.py가 같은 함수를 공유)

모델 입력 포맷 교체 포인트:
    run() 내부의 model.predict() 호출부를 참고.
    다른 모델이 다른 입력 형식을 요구하면 그 부분에 PrepAdapter를 추가하면 됨.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

from tqdm.auto import tqdm

from adapters.input_adapter import InputAdapter, InputAdapterProtocol
from adapters.result_adapter import ResultAdapter, ResultAdapterProtocol
from config import Config


# ── Pipeline 클래스 ───────────────────────────────────────────────────────────

class Pipeline:
    """
    OCR 파이프라인.

    사용 예:
        pipeline = Pipeline(
            model=ocr_model,
            input_adapter=InputAdapter(pdf_dpi=200),
            result_adapter=ResultAdapter(),
        )
        saved = pipeline.run("/data/문서A.pdf", output_dir=Path("./results"), batch_size=8)
    """

    def __init__(
        self,
        model,                               # OCR 모델 (PaddleOCR 등)
        input_adapter: InputAdapterProtocol, # 입력 정규화 어댑터
        result_adapter: ResultAdapterProtocol, # 결과 변환 어댑터
    ):
        """
        어댑터와 모델을 주입받아 저장만 한다.

        model에 타입 힌트를 붙이지 않은 이유:
        - PaddleOCR에는 공식 타입 스텁이 없음
        - 지금 규모에서 ModelProtocol을 별도로 만드는 것은 과도한 추상화
        """
        self.model = model
        self.input_adapter = input_adapter
        self.result_adapter = result_adapter

    def run(
        self,
        source: str | list[str],
        output_dir: Path,
        batch_size: int = 1,   # 기본값 1장 = 단일 추론
        save: bool = True,     # False → JSON 저장 스킵, result dict 리스트 반환
    ) -> list:
        """
        전체 파이프라인 실행.

        실행 순서:
        1. 입력 정규화   — InputAdapter.resolve()로 이미지 경로 목록 획득
        2. 배치 분할     — batch_size 단위로 리스트를 자름
        3. 배치 추론     — model.predict()로 OCR 수행
        4. 결과 변환     — ResultAdapter.convert()로 내부 스키마로 변환
        5. JSON 저장     — save=True일 때만 이미지 1장당 JSON 1개 파일로 저장

        반환:
            save=True  → list[Path] — 저장된 JSON 경로 목록
            save=False → list[dict] — result dict 목록 (저장 없음)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 입력 → 이미지 경로 목록
        #    PDF → JPG 변환, URL 다운로드 등이 여기서 발생
        image_paths = self.input_adapter.resolve(source, output_dir)
        if not image_paths:
            print("처리할 이미지가 없습니다.")
            return []

        print(f"대상 이미지: {len(image_paths)}장  |  배치 크기: {batch_size}")

        # 2. 배치 분할
        #    예) 100장, batch_size=16 → [[0~15], [16~31], ..., [96~99]]
        batches = [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

        # 3~5. 배치 추론 → 변환 → 저장(또는 수집)
        saved: list = []   # save=True → list[Path], save=False → list[dict]
        all_scores: list[float] = []   # 전체 신뢰도 누적 (최종 평균 계산용)
        done = 0
        t_start = time.perf_counter()

        for batch in tqdm(batches, desc="OCR 진행", dynamic_ncols=True):
            # ── 모델 입력 포맷 교체 포인트 ──────────────────────────────
            # 현재: PaddleOCR은 파일 경로 문자열 리스트를 입력받음
            # 변경 시: 다른 모델이 numpy array 등을 요구하면
            #          여기에 PrepAdapter를 추가해 변환하면 됨
            # ────────────────────────────────────────────────────────────
            batch_paths = [str(p) for p in batch]

            # list() 래핑 이유:
            # predict()가 제너레이터를 반환할 수 있음.
            # 제너레이터를 즉시 실체화해야 추론 시간만 정확히 측정되고
            # zip(batch, preds) 순회 시 두 번 소비되는 문제를 방지.
            preds = list(self.model.predict(batch_paths))

            # 배치 신뢰도 수집
            batch_scores = []
            for img_path, pred in zip(batch, preds):
                if hasattr(pred, "to_dict"):
                    raw = pred.to_dict()
                elif hasattr(pred, "keys"):
                    raw = dict(pred)
                else:
                    raw = {}
                scores = raw.get("rec_scores") or raw.get("rec_score") or []
                batch_scores.extend(float(s) for s in scores if s is not None)

                result = self.result_adapter.convert(pred, img_path)
                if save:
                    saved.append(self._save_json(result, img_path, output_dir))
                else:
                    saved.append(result)

            all_scores.extend(batch_scores)
            done += len(batch)

            # 배치 완료 시 속도·신뢰도 출력
            elapsed_now = time.perf_counter() - t_start
            ips_now = done / elapsed_now if elapsed_now > 0 else 0
            remain = (len(image_paths) - done) / ips_now if ips_now > 0 else 0
            batch_conf = sum(batch_scores) / len(batch_scores) if batch_scores else float("nan")
            tqdm.write(
                f"  [{done}/{len(image_paths)}]"
                f"  {ips_now:.2f} img/s"
                f"  배치 avg conf {batch_conf:.3f}"
                f"  남은시간 약 {remain/60:.1f}분"
            )

        # 완료 통계
        elapsed = time.perf_counter() - t_start
        ips = len(image_paths) / elapsed if elapsed > 0 else 0
        total_conf = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        print(f"\n완료: {len(image_paths)}장  |  {elapsed:.1f}초  |  {ips:.2f} img/s  |  전체 avg conf {total_conf:.4f}")
        print(f"저장 경로: {output_dir}")

        return saved

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _save_json(
        self,
        result: dict,
        image_path: Path,
        output_dir: Path,
    ) -> Path:
        """
        결과 dict를 JSON 파일로 저장.

        파일명 규칙:
        - 이미지가 *_images 폴더 안에 있으면 → 형제 *_labels 폴더에 JSON 저장
          예) output_dir/문서A/문서A_images/page_001.jpg
              → output_dir/문서A/문서A_labels/page_001.json

        - 그 외 (단일 이미지, 폴더 입력 등) → 이미지와 같은 위치에 JSON 저장
          예) output_dir/MONO1200706176/page_001.jpg
              → output_dir/MONO1200706176/page_001.json

        - output_dir 외부 경로(단일 이미지 입력 등)는 파일명만 사용
          예) /외부경로/image.jpg → output_dir/image.json
        """
        try:
            rel = image_path.relative_to(output_dir)
            parent = rel.parent
            if parent.name.endswith("_images"):
                # PDF 변환 이미지: *_images → *_labels 폴더로 리다이렉트
                labels_dir = parent.parent / parent.name.replace("_images", "_labels", 1)
                json_path = output_dir / labels_dir / Path(rel.name).with_suffix(".json")
            else:
                json_path = output_dir / rel.with_suffix(".json")
        except ValueError:
            # output_dir 외부 경로(단일 이미지 입력 등)는 파일명만 사용
            json_path = output_dir / image_path.with_suffix(".json").name

        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return json_path


# ── 모델 캐시 + API용 진입점 ──────────────────────────────────────────────────

# 모델 초기화 파라미터 조합별로 PaddleOCR 인스턴스를 보관.
# 같은 조합이 재요청되면 재로딩 없이 캐시된 모델을 반환한다.
# 키: (lang, device, precision, use_doc_orientation_classify,
#       use_textline_orientation, use_doc_unwarping)
_model_cache: dict[tuple, object] = {}


def _get_or_create_model(config: Config) -> object:
    """
    캐시에서 모델을 반환하거나, 없으면 초기화 후 저장.

    PaddleOCR 초기화는 수 초~수십 초 소요되므로
    동일 설정 조합은 반드시 재사용해야 한다.
    """
    from paddleocr import PaddleOCR

    key = (
        config.lang,
        config.device,
        config.precision,
        config.use_doc_orientation_classify,
        config.use_textline_orientation,
        config.use_doc_unwarping,
    )

    if key not in _model_cache:
        _model_cache[key] = PaddleOCR(
            lang=config.lang,
            device=config.device,
            precision=config.precision,
            use_doc_unwarping=config.use_doc_unwarping,
            use_doc_orientation_classify=config.use_doc_orientation_classify,
            use_textline_orientation=config.use_textline_orientation,
        )

    return _model_cache[key]


# Config 기본값을 함수 시그니처 기본값으로 재사용하기 위한 참조
_defaults = Config()


def run_ocr(
    source: str | list[str],
    output_dir: str,
    # ── 런타임 파라미터 (매 호출마다 자유롭게 변경 가능) ──────────────────────
    batch_size: int = _defaults.batch_size,
    pdf_dpi: int    = _defaults.pdf_dpi,
    jpg_quality: int = _defaults.jpg_quality,
    ssl_verify: bool = _defaults.ssl_verify,
    download_timeout: int = _defaults.download_timeout,
    # ── 모델 파라미터 (변경 시 캐시 미스 → 최초 1회 재초기화) ─────────────────
    lang: str      = _defaults.lang,
    device: str    = _defaults.device,
    precision: str = _defaults.precision,
    # ── 문서 방향 설정 (변경 시 캐시 미스 → 최초 1회 재초기화) ──────────────
    use_doc_orientation_classify: bool = _defaults.use_doc_orientation_classify,
    use_textline_orientation:     bool = _defaults.use_textline_orientation,
    use_doc_unwarping:            bool = _defaults.use_doc_unwarping,
) -> list:
    """
    FastAPI / 외부 스크립트에서 OCR 파이프라인을 실행하는 단일 진입점.

    사용 예 (FastAPI):
        from pipeline import run_ocr

        @app.post("/ocr")
        def ocr(req: RunRequest):
            saved = run_ocr(
                source     = req.source,
                output_dir = req.output_dir,
                batch_size = req.batch_size,
                lang       = req.lang,
                device     = req.device,
            )
            return {"saved": [str(p) for p in saved]}

    파라미터 두 종류:
    - 런타임 파라미터: source, output_dir, batch_size, pdf_dpi, jpg_quality 등
      → 매 호출마다 다른 값을 줄 수 있음. 모델 재로딩 없음.
    - 모델 파라미터: lang, device, precision, use_* 플래그
      → 변경 시 새 조합으로 최초 1회 PaddleOCR 초기화 발생 (수 초 소요).
        이후 동일 조합은 캐시에서 즉시 반환.

    문서 방향 설정 가이드:
    - PDF → JPG 변환 이미지, 정립된 스캔본: 모두 False (기본값) — 속도 최우선
    - 방향 불확실한 사진·모바일 촬영 이미지: use_doc_orientation_classify=True
    - 세로쓰기(일본어·한문 고서): use_textline_orientation=True
    - 구겨지거나 굴곡 있는 문서: use_doc_unwarping=True (가장 느린 단계)

    반환:
        output_dir 지정 → list[Path] — 저장된 JSON 경로 목록
        output_dir=''  → list[dict] — result dict 목록 (저장 없음, dry-run)
    """
    dry_run = not output_dir   # output_dir='' → 저장 스킵

    config = Config(
        lang=lang,
        device=device,
        precision=precision,
        batch_size=batch_size,
        pdf_dpi=pdf_dpi,
        jpg_quality=jpg_quality,
        ssl_verify=ssl_verify,
        download_timeout=download_timeout,
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_textline_orientation=use_textline_orientation,
        use_doc_unwarping=use_doc_unwarping,
    )

    # 모델은 캐시에서 반환 (재로딩 없음)
    # InputAdapter/ResultAdapter는 런타임 파라미터를 반영해 매번 새로 생성
    pipeline = Pipeline(
        model=_get_or_create_model(config),
        input_adapter=InputAdapter(
            pdf_dpi=config.pdf_dpi,
            jpg_quality=config.jpg_quality,
            ssl_verify=config.ssl_verify,
            download_timeout=config.download_timeout,
        ),
        result_adapter=ResultAdapter(),
    )

    if dry_run:
        # PDF→JPG 변환·URL 다운로드가 필요한 경우를 위해 임시 폴더 사용.
        # 이미지 파일·폴더 입력은 work_dir을 실제로 쓰지 않음.
        work_dir = Path(tempfile.mkdtemp(prefix="paddle_ocr_"))
        try:
            return pipeline.run(source, work_dir, batch_size=batch_size, save=False)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    return pipeline.run(source, Path(output_dir), batch_size=batch_size, save=True)


# ── 팩토리 함수 ───────────────────────────────────────────────────────────────

def build_pipeline(config: Config) -> Pipeline:
    """
    Config 설정으로 Pipeline을 조립하는 팩토리 함수.

    이 함수가 존재하는 이유:
    - main.py(CLI)와 api.py(FastAPI) 양쪽에서 동일한 방법으로 파이프라인을 만들 수 있음
    - 파이프라인 조립 로직을 한 곳에 집중시켜 중복 제거
    - PaddleOCR 초기화가 여기서 한 번만 일어남
      (GPU 모델 로딩은 수 초~수십 초 소요 → 한 번 초기화 후 run()을 여러 번 호출)
    """
    from paddleocr import PaddleOCR

    model = PaddleOCR(
        lang=config.lang,
        device=config.device,
        precision=config.precision,
        use_doc_unwarping=config.use_doc_unwarping,
        use_doc_orientation_classify=config.use_doc_orientation_classify,
        use_textline_orientation=config.use_textline_orientation,
    )

    return Pipeline(
        model=model,
        input_adapter=InputAdapter(
            pdf_dpi=config.pdf_dpi,
            jpg_quality=config.jpg_quality,
            ssl_verify=config.ssl_verify,
            download_timeout=config.download_timeout,
        ),
        result_adapter=ResultAdapter(),
    )


if __name__ == "__main__":
    source     = "/root/project/data1/vision/data/국회도서관/pdf2jpg_176/000010.jpg"
    output_dir = ""           # '' → dry-run (저장 없음), 경로 지정 시 JSON 저장
    batch_size = 1
    device     = "gpu:0"      # int 아닌 문자열
    lang       = "korean"
    use_textline_orientation = False   # True: 세로쓰기, False: 가로

    result = run_ocr(
        source,
        output_dir,
        batch_size = batch_size,
        lang       = lang,
        device     = device,
        use_textline_orientation = use_textline_orientation,
    )
    print(result)
