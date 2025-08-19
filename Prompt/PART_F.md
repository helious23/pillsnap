# Part F — FastAPI 통합 (/health /version /predict /batch /reload + 보안/CORS/성능/로그)

# 목적:

# 1) 사용자 제어 Two-Stage 파이프라인으로 약품 이미지에서 edi_code를 추출하여 API로 서빙한다.

# 2) 보안(X-API-Key), CORS, 파일 검증, 리소스 한도를 명시적으로 적용한다.

# 3) 성능(워밍업/모드별 최적화/검출기 지연 로딩)과 운영(로그/상태/리로드)을 고려한 구조를 제공한다.

[전제/경로/규칙]

- 코드 루트: /home/max16/pillsnap
- 모든 경로는 /mnt/…(WSL)만 사용(C:\ 금지).
- 편집 위치는 자유(맥/윈도우/원격). **실행은 WSL 기준**으로 한다.
- 체크포인트/로그/산출물은 **/mnt/data**(WSL 디스크)에 저장(속도/안정성↑).
- .env(.env.example): API_KEY, LOG_LEVEL, DETECTION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, CORS_ALLOW_ORIGINS 사용.
- 기본 엔진 우선순위: **ONNX → Torch**(ONNX 미존재/실패 시 Torch 폴백).
- 단일 우선 파이프라인: single(직접 분류) → combo(검출→크롭→분류) → auto(최소 사용).
- 이미지 크기/정규화: **src/infer.py의 단일 전처리 함수 공유 강제**(검출: 640, 분류: 384).

[이 파트에서 구현/수정할 파일]

1. src/api/main.py # FastAPI 앱, 라우팅, 미들웨어, 수명 훅(startup/shutdown)
2. src/api/security.py # .env 로드(Settings), X-API-Key 인증 데코레이터, CORS, 업로드/확장자 정책, (간이) rate-limit
3. src/api/schemas.py # Pydantic v2 스키마: HealthResp, VersionResp, PredictReq/Resp, BatchReq/Resp, ErrorResp
4. src/api/service.py # SingleFirstModelManager 싱글톤: 분류 모델(주력)+검출 모델(선택적) 로드/캐시/워밍업/predict/reload, 약품 메타데이터 매핑
5. src/infer.py # (재사용) preprocess(), infer_torch(), infer_onnx() — Part E에서 정의한 것 사용 (전처리 단일화 필수)
6. scripts/run_api.sh # uvicorn 실행 스크립트(--tmux | --no-tmux), 로그 리다이렉트

──────────────────────────────────────────────────────────────────────────────
F-1) 보안/설정(.env) — security.py

- Settings(pydantic-settings):
  class Settings:
  API_KEY: str
  LOG_LEVEL: Literal["debug","info","warning","error","critical"] = "info"
  DETECTION_MODEL_PATH: str | None = None # 검출 모델 경로 (우선 로드)
  CLASSIFICATION_MODEL_PATH: str | None = None # 분류 모델 경로 (우선 로드)
  DRUG_METADATA_PATH: str | None = None # 약품 메타데이터 매핑 파일 경로
  CORS_ALLOW_ORIGINS: str = "http://localhost:3000,https://pillsnap.co.kr,https://api.pillsnap.co.kr"
  MAX_UPLOAD_MB: int = 20
  ALLOWED_EXTS: list[str] = [".jpg",".jpeg",".png",".bmp",".webp"]
  RATE_LIMIT_PER_MIN: int = 60 # 초간단 per-process 카운터
- get_settings(): 단일 인스턴스 캐시.
- CORS:
  CORSMiddleware(origins = split by comma, allow_methods=["*"], allow_headers=["*"])
- X-API-Key 미들웨어/디펜던시:
  - /health는 공개 (보안 정책), /version, /predict, /batch, /reload는 보호 필수
  - 인증 실패 시 401(WWW-Authenticate 헤더는 생략 가능).
- 파일 검증:
  - Content-Type: image/\* 만 허용(멀티파트 검사)
  - 확장자 화이트리스트 검사
  - 파일 크기 제한: StreamingBody 길이를 검사하거나 read() 후 len() 체크(20MB 기본).
- (간이) rate-limit:
  - in-memory {api_key: deque[timestamps]}로 1분당 N회 제한. 초과 시 429.

──────────────────────────────────────────────────────────────────────────────
F-2) 스키마 — schemas.py (Pydantic v2)

- HealthResp:
  { "status":"ok", "device":"cuda|cpu", "uptime_sec": float, 
  "detection_engine":"onnx|torch", "classification_engine":"onnx|torch", 
  "models_loaded": {"detection": bool, "classification": bool} }
- VersionResp:
  { "app":"pillsnap-ml", "git_sha": "...|nogit", 
  "detection_model_path":".../detection-....onnx|best.pt",
  "classification_model_path":".../classification-....onnx|best.pt",
  "drug_metadata_loaded": bool, "num_classes":int, "exported_at":"UTC|None" }
- DetectionResult:
  { "bbox": [x1, y1, x2, y2], "confidence": float, "class_id": int, "drug_metadata": dict }
- PredictResp (Two-Stage):
  { "detections": [DetectionResult, ...], "detection_count": int,
  "detection_engine":"onnx|torch", "classification_engine":"onnx|torch", 
  "time_ms": {"detection": float, "classification": float, "total": float}, 
  "input_shape": [N,C,H,W], "version_tag": "UTC-SHA",
  "mode_requested": "single|combo", "mode_used": "single|combo",
  "confidence": float }
- BatchResp:
  { "results": [PredictResp-lite...], "detection_engine":"...", "classification_engine":"...", 
  "time_ms": float, "count": int, "total_detections": int }
- ErrorResp:
  { "detail": str, "code": str }

(검출/세그는 후속 확장: F-Det에서 별도 스키마 추가)

──────────────────────────────────────────────────────────────────────────────
F-3) 서비스 레이어 — service.py (SingleFirstModelManager)

- 싱글톤(ModelManager.get()):
  내부 상태:
  self.classification_engine: "onnx"|"torch"  # 주력 (항상 로드)
  self.detection_engine: "onnx"|"torch"      # 조합용 (즉시 로드)
  self.classification_model: torch.nn.Module | None
  self.classification_onnx_session: ort.InferenceSession | None
  self.detection_model: YOLO | torch.nn.Module | None
  self.detection_onnx_session: ort.InferenceSession | None
  self.device: torch.device
  self.drug_metadata: dict[int, dict] # class_id → complete_metadata
  self.version_tag: str | None
  self.started_at: float
- 로드 우선순위 (직렬 실행 + 지연 로딩):
  Classification (즉시 로드 - 주력 모델):
  1. ENV CLASSIFICATION_MODEL_PATH 지정 시 우선  
  2. cfg.export.out_dir의 classification-*.onnx 최신
  3. checkpoints/classification_best.pt
  Detection (지연 로딩 - combo 모드 시만):
  1. Load Once Guard: 첫 combo 요청에서만 로드 (뮤텍스로 동시 로드 방지)
  2. Idle TTL Reaper: 마지막 combo 후 10분 지나면 언로드 (백그라운드 타이머)
  3. Hysteresis: 로드/언로드 사이 2분 최소 체류 (스래싱 방지)
  4. Optional: --prewarm_combo 플래그로 스타트업 사전 로딩
  Drug Metadata:
  1. ENV DRUG_METADATA_PATH 지정 시 우선
  2. cfg.data.drug_metadata_file
- onnx 로드:
  - select_onnx_providers(): CUDA EP → CPU EP(Part E 규칙)  
  - Classification: timm ONNX 세션 생성 + warmup(1x3x384x384) - 즉시
  - Detection: YOLOv11 ONNX 세션 생성 + warmup(1x3x640x640) - 즉시
- torch 로드:
  - Classification: timm.create_model() + state_dict 로드, model.eval() - 즉시
  - Detection: YOLO(model_path) 로드, model.eval() - 즉시
  - AMP off(기본), warmup 수행
- 공통 워밍업 (차별화):
  - 분류: preprocess_classification(img, 224) → 예열 추론 - 필수 (startup 시)
  - 검출: preprocess_detection(img, 640) → 예열 추론 - 지연 (combo 첫 요청 시)
  - 예열 후 각 단계별 latency 로그
- predict(image_bytes | PIL.Image | np.ndarray, mode="single", conf_threshold=0.3):
  - mode="single": 직접 분류 → class_id + drug_metadata (90% 케이스)
  - mode="combo": 직렬 실행 (YOLO 지연 로딩 → 검출 → crop → 분류 → 매핑)
    1. Detection 모델 로딩 체크 (미로드 시 lazy load)
    2. YOLO 추론 → bbox 리스트 
    3. 각 bbox crop → 분류 모델 추론 (순차)
    4. drug_metadata 매핑 (배치)
  - 메모리 관리: 
    - Classification 모델: 항상 GPU 상주 (3-5GB)
    - Detection 모델: Load-Once + Idle-TTL + Hysteresis 3규칙 적용
  - 시간 측정: classification_ms (single), detection_ms + classification_ms (combo)
  - 반환 형태: 
    - single: {"class_id": int, "confidence": float, "drug_metadata": dict, "hint": str|None}
    - combo: {"detections": [{"bbox": [x1,y1,x2,y2], "class_id": int, "confidence": float, "drug_metadata": dict}]}
  - 메타데이터 매핑 실패 시 간단 폴백: {"error": "metadata_not_found", "class_id": class_id}, 경고 로그
- reload(detection_path: str=None, classification_path: str=None, drug_metadata_path: str=None):
  - 지정된 모델만 리로드, 워밍업 후 엔진 갱신
  - 성공 시 현재 버전/경로 반환

──────────────────────────────────────────────────────────────────────────────
F-4) FastAPI 앱 — main.py

- 앱 생성:
  app = FastAPI(title="pillsnap-ml", version="1.0.0")
  로거 레벨: settings.LOG_LEVEL 반영
  CORS/에러 핸들러/요청 ID 미들웨어(간단히 uuid4) 추가
- 수명 훅(startup):
  - config.yaml 로드
  - SingleFirstModelManager.get().load_models(cfg, settings)
  - drug_metadata 로드 및 검증, classification 모델 우선 로드, detection 지연 로딩 설정
- 라우트:
  GET /health (공개): HealthResp
  GET /version (보호 필수): VersionResp
  POST /predict (보호): multipart/form-data { image: File }, query: mode="single"|"combo", conf_threshold(기본 0.3)
    # mode 결정 우선순위: 1) 쿼리 mode 파라미터 (최상위), 2) config.data.default_mode (빈값일 때만), 3) auto_fallback=false (항상 비활성)
  POST /batch (보호): multipart/form-data { images: File[List] }, mode="single"(기본), 개수 제한(≤16)
  POST /reload (보호+관리): JSON { "detection_path": str|None, "classification_path": str|None, "drug_metadata_path": str|None }
- 유효성:
  - 파일 크기/확장자/Content-Type 검사(초과/불일치 시 422 or 413)
  - batch는 총 용량 합산 체크(예: 100MB)
- 응답:
  - mode별 차별화: 
    * single(class_id, confidence, drug_metadata, hint): 단일 약품 직접 분류 결과
    * combo(detections list with metadata): 여러 약품 검출 후 분류 결과
  - classification_ms, detection_ms(combo만), total_ms, engine 정보, version_tag 포함
  - 오류는 ErrorResp 스키마로 통일, 필요한 경우 400/401/413/415/422/429/500
  - 모드 결정 규칙: 쿼리 mode가 최상위, default_mode는 빈값일 때만, auto_fallback=false 항상
- 로깅:
  - 각 요청마다 request_id, client_ip, method, path, size, mode, classification_engine, detection_engine, total_latency(ms) 기록
  - 예외는 stacktrace 요약 + code

──────────────────────────────────────────────────────────────────────────────
F-5) 엔드포인트 명세(요약)

- GET /health (공개)
  200: {"status":"ok","device":"cuda","detection_engine":"onnx","classification_engine":"onnx",
  "models_loaded":{"detection":true,"classification":true},"uptime_sec":12.3}
- GET /version (보호 필수)
  200: {"app":"pillsnap-ml","git_sha":"abc1234","detection_model_path":"...detection.onnx",
  "classification_model_path":"...classification.onnx","drug_metadata_loaded":true,
  "num_classes":5000,"exported_at":"2025-08-10T14:30:12Z"}
- POST /predict (보호)
  요청: multipart/form-data, image=@/path/img.jpg, headers: X-API-Key
  쿼리: mode="single"(기본)|"combo", conf_threshold=0.3
  200 (single): {"class_id":1234,"confidence":0.95,"drug_metadata":{"di_edi_code":"12345","dl_name":"게루삼정 200mg/PTP",...},"hint":null,"classification_engine":"onnx",
  "time_ms":{"classification":15.2,"total":15.2},"version_tag":"UTC-SHA"}
  200 (combo): {"detections":[{"bbox":[x1,y1,x2,y2],"confidence":0.95,"class_id":1234,"drug_metadata":{"di_edi_code":"12345",...}}],
  "detection_count":1,"detection_engine":"onnx","classification_engine":"onnx",
  "time_ms":{"detection":15.2,"classification":8.7,"total":23.9}}
- POST /batch (보호)
  요청: images=@img1, images=@img2, mode="single"(기본) ... (최대 16개)
  200: BatchResp {results:[{mode: "single", result: {class_id, confidence, drug_metadata, hint}},...], 
  count:n, classification_engine:"onnx", detection_engine:"onnx"|null, time_ms:...}
- POST /reload (보호+관리자)
  요청: {"detection_path": "/path/detection.onnx", "classification_path": "/path/classification.onnx", "drug_metadata_path": "/path/drug_metadata.json"}
  200: {"ok":true,"detection_engine":"onnx","classification_engine":"onnx","version_tag":"UTC-SHA"}

──────────────────────────────────────────────────────────────────────────────
F-6) 성능/안정성 정책

- 배치 추론:
  - /batch에서 N개 이미지를 mode별 개별 처리, 최대 N=16 제한
  - single 모드: 분류만 수행으로 빠른 처리 (기본)
  - combo 모드: 검출 → crop → 분류 순차 처리
- 워밍업:
  - startup에서 classification(384x384) 필수 예열, detection(640x640) 지연 예열
- 시간 제한:
  - uvicorn workers=1(단일 GPU 가정), keep-alive/timeouts 보수적으로
- 예외·폴백:
  - classification onnx 실패 시 torch로 임시 폴백
  - detection은 지연 로딩으로 combo 모드 첫 사용 시 로드/폴백
  - 지속 실패 시 500 응답
- CORS 최소화:
  - settings.CORS_ALLOW_ORIGINS만 허용. 운영 시 프런트 도메인 외 제거 권장.

──────────────────────────────────────────────────────────────────────────────
F-7) 스크립트 — scripts/run_api.sh

- 옵션:
  --tmux : tmux 세션(pillsnap_api)로 백그라운드 실행 후 attach 안내
  --no-tmux : 포그라운드 실행(기본)
- 로직:
  set -euo pipefail
  VENV="$HOME/pillsnap/.venv"; ROOT="/home/max16/pillsnap"
  source "$VENV/bin/activate" && cd "$ROOT"
  export $(grep -v '^#' .env | xargs -d '\n' -I {} echo {}) || true # .env 있으면 로드
  # LOGDIR 계산 (yq 없거나 null이면 ./logs로 폴백)
  YQ_VAL=$(yq '.paths.exp_dir' config.yaml 2>/dev/null || echo "")
  if [ -z "$YQ_VAL" ] || [ "$YQ_VAL" = "null" ]; then LOGDIR="./logs"; else LOGDIR="${YQ_VAL}/logs"; fi
  mkdir -p "$LOGDIR"
  CMD="uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 5"
  case "${1:---no-tmux}" in
  --tmux)
  TMUX_NAME="pillsnap_api"
  tmux new -d -s "$TMUX_NAME" "$CMD >> $LOGDIR/api.out 2>> $LOGDIR/api.err"
      echo "[OK] API in tmux session: $TMUX_NAME"; echo "  tmux attach -t $TMUX_NAME"
      ;;
    --no-tmux|*)
      eval "$CMD"
  ;;
  esac

──────────────────────────────────────────────────────────────────────────────
F-8) 예시 호출(curl)

# 헬스

$ curl http://localhost:8000/health

# 예측(단일) - 직접 분류 → edi_code (기본 모드)

$ curl -H "X-API-Key: CHANGE_ME_STRONG_RANDOM" \
 -F "image=@pill001.jpg" \
 "http://localhost:8000/predict?mode=single&conf_threshold=0.3"

# 예측(조합) - 검출 + 분류 → 다중 edi_codes

$ curl -H "X-API-Key: CHANGE_ME_STRONG_RANDOM" \
 -F "image=@pills_combo.jpg" \
 "http://localhost:8000/predict?mode=combo&conf_threshold=0.3"

# 배치(여러 파일) - 기본 단일 모드

$ curl -H "X-API-Key: CHANGE_ME_STRONG_RANDOM" \
 -F "images=@pill001.jpg" \
 -F "images=@pill002.jpg" \
 "http://localhost:8000/batch?mode=single"

# 리로드(새 모델들로 교체)

$ curl -H "X-API-Key: CHANGE_ME_STRONG_RANDOM" \
 -H "Content-Type: application/json" \
 -d '{"detection_path":"/mnt/data/exp/exp01/export/detection-20250810-143012-ab12cd3.onnx", "classification_path":"/mnt/data/exp/exp01/export/classification-20250810-143012-ab12cd3.onnx", "drug_metadata_path":"/mnt/data/exp/exp01/drug_metadata.json"}' \
 http://localhost:8000/reload

──────────────────────────────────────────────────────────────────────────────
F-9) 테스트(최소 스모크) — tests/test_api_min.py

- 조건: detection + classification 모델 최소 1개씩 로드 (.env 경로 권장)
- 시나리오:
  1. /health 200, detection/classification 모델 로드 상태 검증
  2. /version (보호/공개 설정에 맞게) 200, drug_metadata_loaded 확인
  3. /predict: 샘플 약품 이미지 1장으로 200, DetectionResult 스키마(bbox/confidence/class_id/drug_metadata 존재)
  4. /predict: X-API-Key 누락 → 401/403
  5. /batch: 2~3장 업로드로 200, detection_count/total_detections 일치
  6. /reload: 유효 detection/classification onnx로 200 → /version 갱신 확인

──────────────────────────────────────────────────────────────────────────────
F-10) 운영 팁

- API Key는 .env에 강한 랜덤값(예: `openssl rand -hex 32`) 부여, 주기 교체.
- 업로드 제한/확장자 화이트리스트는 필수. 외부 공개 전 파일검증 로그를 살펴 성능 영향 확인.
- uvicorn은 단일 워커(단일 GPU). 여러 GPU/인스턴스면 Cloudflare Load Balancer로 수평 확장.
- Cloudflare Tunnel 연동은 **Part G**에서 서비스화(trycloudflare → 영구 터널 `api.pillsnap.co.kr`).

## 🎯 **PART_F 핵심 업데이트 완료**

### ✅ **사용자 제어 기반 Two-Stage API 서빙**
- **Frontend 선택**: 사용자가 mode 파라미터로 직접 선택 (single/combo)
- **Single 모드** (90% 케이스): 직접 EfficientNetV2-S 분류 (384px) → edi_code 반환
- **Combo 모드** (명시적 선택): YOLOv11m 검출 (640px) → 크롭 → 분류 (384px) → 다중 edi_code 반환
- **복잡도 감소**: 자동 판단 로직 완전 제거, 단순한 모드 분기 구조

### ✅ **FastAPI 서비스 아키텍처**  
- **/predict**: 단일 이미지 + mode 파라미터 → edi_code(s) 반환
- **/batch**: 다중 이미지 배치 처리 + 모드 선택 (RTX 5080 최적화)
- **/health**: 헬스체크 (모델 로딩 상태 포함)
- **/reload**: 무중단 모델 교체 (hot-reload)
- **사용자 컨트롤**: Frontend에서 명확한 모드 선택권 제공

### ✅ **보안 & 최적화**
- **인증**: X-API-Key 헤더 기반 인증
- **CORS**: pillsnap.co.kr 도메인 화이트리스트
- **성능**: 모델 메모리 상주, 배치 추론 최적화
- **로깅**: 요청/응답/에러 상세 로깅

**✅ PART_F 완료: 사용자 제어 Two-Stage FastAPI 서빙**
