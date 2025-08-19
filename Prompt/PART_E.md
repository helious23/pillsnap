# Part E — Export(ONNX) · 버전 태깅 · Torch↔ONNX 동등성 검증 · .env 연동 · 테스트

# 목적:

# 1) Two-Stage 학습 산출물(detection.pt + classification.pt)을 각각 ONNX로 내보낸다.

# 2) 통합 추론 파이프라인(detection → crop → classification → edi_code)을 구성한다.

# 3) Torch vs ONNX 동등성을 정량 검증(detection: mAP 차이≤0.01, classification: MSE≤1e-6)한다.

# 4) 버전 태깅(UTC+git SHA)과 export_report.json을 남긴다.

# 5) .env의 MODEL_PATH를 통해 API/런타임 모델 고정을 지원한다.

[전제/경로/규칙]

- 코드 루트: /home/max16/pillsnap
- exp_dir: **/home/max16/ssd_pillsnap/exp/exp01** (SSD 이전 완룼)
- 모든 데이터 경로는 **SSD 기반** (/home/max16/ssd_pillsnap/)만 사용. HDD 경로(/mnt/data/) 백업용.
- 기본 작업은 검출+분류(Detection+Classification, Two-Stage). 약품 검출 후 분류하여 edi_code 반환.

[필수 의존성(WSL)]

- Python 패키지: torch, timm, ultralytics, onnx, onnxruntime **(GPU가 있으면 onnxruntime-gpu)**, numpy
- CLI: yq (mikefarah 버전; YAML 파서)
  - 점검: `yq --version` / 미설치 시 설치 안내 출력 후 종료
- 권장: pip install tensorboard (E-10의 TBWriter 대비)

[이 파트에서 구현/수정할 파일]

1. scripts/export_onnx.sh # Two-Stage 원클릭 export + 비교 리포트
2. src/models/detector_yolo11m.py + src/models/classifier_efficientnetv2_s.py # export_detection_onnx(), export_classification_onnx(), build_detection_model(), build_classification_model(), count_params(), prepare_version_tag()
3. src/infer.py # 통합 추론 파이프라인; detection → crop → classification → edi_code 매핑; Torch/ONNX 런처(select_onnx_providers 포함)
4. src/utils.py # utc_timestamp(), get_git_sha(), save_json(), ensure_dir(), list_checkpoints(), find_best_checkpoint()
5. tests/test_export_compare.py # 동등성 검증 테스트(detection: mAP, classification: MSE/Top-1)

──────────────────────────────────────────────────────────────────────────────
E-1) Two-Stage Export 사양(강제)

- 입력 ckpt: 
  - Detection: {exp_dir}/checkpoints/detection_best.pt 
  - Classification: {exp_dir}/checkpoints/classification_best.pt
- 출력 onnx:
  - Detection: {exp_dir}/export/detection-<UTC>-<gitsha|nogit>.onnx
  - Classification: {exp_dir}/export/classification-<UTC>-<gitsha|nogit>.onnx
  - UTC fmt: %Y%m%d-%H%M%S (예: 20250810-143012)
- opset: 17 | dynamic_axes: True (호환성 검증됨)
  - Detection input: [N,3,640,640] → N:batch, H:height, W:width (동적)
  - Detection output: [N,boxes,6] (x1,y1,x2,y2,conf,class_id) (동적)
  - Classification input: [N,3,384,384] → N:batch (동적)
  - Classification output: [N,num_classes] (동적 N)
  
# ONNX 호환성 검증 (opset 17 지원 확인)
onnx_compatibility:
  yolov11m:
    source: "Ultralytics 공식 문서"
    opset_support: 17
    verification: "공식 이슈 로그에서 opset 17 내보내기 성공 사례 확인"
    fallback: "opset 18 시도 후 ORT 버전 업그레이드 검토"
  efficientnetv2_s:
    source: "timm 라이브러리 + PyTorch 공식 Exporter"
    opset_support: 17
    verification: "timm 측 ONNX Export 지원, PyTorch 공식 문서 기준 opset 17 지원"
    error_handling: "특정 연산 미지원 시 PyTorch가 명시적 오류 메시지 제공"
  onnx_runtime:
    opset_support: "17 포함 폭넓은 버전 지원"
    providers: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    loading_safety: "검증된 호환성으로 로딩 측면에서 안전"

- Detection Export:
  - YOLOv11 모델 → ultralytics 내장 export 기능 사용
  - yolo_model.export(format="onnx", opset=17, dynamic=True)
  - 가능하면 NMS 이전의 raw boxes/scores/class 출력을 사용하고, 후처리는 공통 Python NMS로 수행

- Classification Export:
  - timm.create_model(backbone, pretrained=False, num_classes)
  - torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["logits"],
    opset_version=17, do_constant_folding=True,
    dynamic_axes={"input": {0:"batch"}, "logits": {0:"batch"}}
    )

- 예외/대응:
  - Unsupported op → 친절 로그 + "opset 18 시도" 안내(기본은 17 고정)
  - half/bfloat export 이슈 시: export 전 model.float()로 변환(런타임은 ORT가 최적화)
  
- 부가 산출물:
  - {exp_dir}/export/export_report.json (아래 E-3 스키마)
  - latest_detection.onnx, latest_classification.onnx 심볼릭 링크 생성
  - 통합 파이프라인 스크립트: pillsnap_pipeline.py (detection → classification 연결)

──────────────────────────────────────────────────────────────────────────────
E-2) Torch↔ONNX 동등성 검증(강제)

- 샘플 추출(편향 최소화): config.export.compare에 따라 **계층적(stratified) 샘플링** 수행
  - mode: "smoke" | "coverage" | "full"
    - smoke: 빠른 스모크 테스트. sample_count(기본 32)개 무작위 추출
    - coverage: **클래스 균형 샘플링**. per_class_k(기본 1)씩 수집하되,
      min_classes(기본 1000) 이상을 충족하고 max_total(기본 5000) 내에서 제한
    - full: val 전체 사용(스트리밍 배치 추론, 메모리 제약 없음)
  - stratify_by: ["class"] (필수)
  - (선택) hardness_bins: [0.0,0.3,0.6,0.85,1.0]에서 **결정 경계 근처 샘플**을 bin별 추가(hard_per_bin)
- 전처리: src/data.py의 **검증 파이프라인과 동일**(Resize/CenterCrop/Normalize)
- 두 엔진 실행:
  - Torch: model(x).softmax(dim=1) → prob_t
  - ONNX: onnxruntime.InferenceSession → prob_o
- 실용적 통계 기준(coverage/full):
  - classification: MSE(mean) ≤ 1e-4, MSE p99 ≤ 5e-4, Top-1 mismatch_rate ≤ 1% (0.01)
  - fp16 환경 완화: MSE(mean) ≤ 5e-4, Top-1 mismatch_rate ≤ 2% (0.02)
  - detection: mAP Δ≤0.01(0.5 & 0.5:0.95) + p95 IoU Δ≤0.01
  - 보고: per-class mismatch Top-10, 경계 사례 로그
- 실패 시:
  - 허용 오차 상향 가이드(예시: 1e-5)와 원인 후보(op, 정규화 불일치, 변환 차이)를 로그
  - export_report.json에 실패 이유 기록 후 **비정상 종료 코드(1)** 반환
  - 검출 비교는 Torch/ONNX 모두 공통 NMS 적용 결과를 기준으로 mAP Δ≤0.01을 검증

──────────────────────────────────────────────────────────────────────────────
E-3) export_report.json 스키마(강제)
{
"backbone": "efficientnet_v2_s",
"num_classes": 5000,
"ckpt_path": "/mnt/data/exp/exp01/checkpoints/best.pt",
"onnx_path": "/mnt/data/exp/exp01/export/model-20250810-143012-ab12cd3.onnx",
"exported_at_utc": "2025-08-10T14:30:12Z",
"git_sha": "ab12cd3" | "nogit",
"input_shape": [1,3,224,224],
"opset": 17,
"dynamic_axes": true,
"params_million": 21.32,
"providers": ["CUDAExecutionProvider","CPUExecutionProvider"],
"postprocess": {
  "nms": {
    "conf_threshold": 0.3,
    "iou_threshold": 0.5,
    "max_detections": 100,
    "class_agnostic": false
  },
  "applied_in": "python_common"
},
"compare": {
  "mode": "coverage",
  "sample_policy": "stratified",
  "sample_count": 1200,
  "per_class_k": 1,
  "min_classes": 1000,
  "max_total": 5000,
  "sampled_classes": 1187,
  "sampled_images": 1200,
  "mse_mean": 7.4e-7,
  "mse_p99": 3.2e-6,
  "top1_match_rate": 1.0,
  "top1_mismatch_count": 0,
  "failed": false,
  "failure_reason": null,
  "deltas": {"map_0_5": 0.0, "map_0_5_0_95": 0.0},
  "notes": "stratified by class; hardness bins enabled"
}
}

──────────────────────────────────────────────────────────────────────────────
E-4) onnxruntime 프로바이더 선택 정책(select_onnx_providers)

- 기본 우선순위: CUDAExecutionProvider → CPUExecutionProvider
- CUDA EP 옵션 예:
  provider_options=[{"cudnn_conv_use_max_workspace": 1}]
- 세션 옵션:
  sess_options.graph_optimization_level = ORT_ENABLE_ALL
  - CPU 경로: sess_options.intra_op_num_threads=0, inter_op_num_threads=0 (자동)
  - 메모리/스레드 제어는 ORT 버전에 따라 일부 옵션명이 다를 수 있으므로 로그로 실제 적용값을 남긴다.
- 예외:
  - GPU가 없어 CUDA EP 생성 실패 시 자동 폴백(CPU EP), 로그에 명시
  - 텐서RT EP는 본 프로젝트 기본 비활성(필요 시 별도 안내)

──────────────────────────────────────────────────────────────────────────────
E-5) .env 연동 (MODEL_PATH)

- .env(.env.example 기반)에 MODEL_PATH 키를 두어 **API/런타임에서 우선 사용**:
  MODEL_PATH=/mnt/data/exp/exp01/export/model-20250810-143012-ab12cd3.onnx
- src/api/service.py의 ModelManager는 로딩 우선순위를 다음과 같이 가져간다:
  1. ENV: MODEL_PATH (존재/가독성 검사)
  2. config.export.out_dir 내 최신 onnx (latest.onnx 또는 가장 최근 파일)
  3. 없으면 Torch ckpt(best.pt) 경로 사용(엔진 Torch로 폴백)
- /reload 엔드포인트는 JSON의 "model_path"로 강제 전환하고 버전 업데이트

──────────────────────────────────────────────────────────────────────────────
E-6) scripts/export_onnx.sh (완전 로직 지시)

- set -euo pipefail
- 사전 점검: `command -v yq >/dev/null || { echo "[E-6] yq 미설치: https://github.com/mikefarah/yq 참고"; exit 1; }`
- (옵션) OPSET 환경변수 지원: `OPSET=${OPSET:-17}` # 기본 17, 실패 시 18 시도로 안내 로그
- VENV="$HOME/pillsnap/.venv"; ROOT="/home/max16/pillsnap"
- source "$VENV/bin/activate" && cd "$ROOT"
- CFG="config.yaml"
- EXP_DIR=$(yq '.paths.exp_dir' "$CFG")
- CKPT="$EXP_DIR/checkpoints/best.pt"
- OUT_DIR=$(yq '.export.out_dir' "$CFG")
- 샘플 수: N=$(yq '.export.compare.sample_count' "$CFG")
- 사전 체크: CKPT/OUT_DIR 존재 확인, OUT_DIR 없으면 생성
- python - <<'PY'

# 환경변수 OPSET을 읽어 기본 opset_version을 설정하고, 실패 시 opset 18 재시도 안내만 로그로 남길 것.

# 1) cfg 로드, 2) 모델 빌드+ckpt 로드, 3) onnx export, 4) Torch/ONNX 비교, 5) export_report.json 저장

# - 실패 시 sys.exit(1)

# - 성공 시 onnx 경로 echo

PY

- 종료 코드 분기: 0이면 SUCCESS(onnx 경로 출력), 아니면 FAIL(로그 경로 안내)

──────────────────────────────────────────────────────────────────────────────
E-7) src/model.py에 추가할 함수(구현 지시)

- build_model(backbone, num_classes, channels_last):
  m = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
  if channels_last: m = m.to(memory_format=torch.channels_last)
  return m
- prepare_version_tag():
  sha = get_git_sha() # 없으면 "nogit"
  ts = utc_timestamp() # UTC 문자열
  tag = f"{ts}-{sha[:7] if sha!='nogit' else 'nogit'}"
  return tag
- count_params(model) -> float(M): sum(p.numel() for p in model.parameters())/1e6
- export_onnx(cfg, ckpt_path, out_dir):
  1. build_model() → load_state_dict(엄격 검사/불일치 시 친절 로그)
  2. model.eval().float(); device = cuda or cpu
  3. dummy = torch.randn(*cfg.export.input_shape, device=device)
  4. torch.onnx.export(..., opset=17, dynamic_axes=..., do_constant_folding=True)
  5. 파일명 = f"model-{prepare_version_tag()}.onnx"
  6. 반환: onnx_path, meta(dict: params_million, opset, input_shape)

──────────────────────────────────────────────────────────────────────────────
E-8) TensorRT 최적화 Export (RTX 5080 특화)

**TensorRT 최적 프로바이더 설정**:
```python
def select_onnx_providers_tensorrt_first():
    """TensorRT → CUDA → CPU 우선순위로 프로바이더 선택"""
    available_providers = ort.get_available_providers()
    
    # TensorRT 우선 시도
    if "TensorrtExecutionProvider" in available_providers:
        try:
            # RTX 5080 최적화 설정
            tensorrt_options = {
                'device_id': 0,
                'trt_max_workspace_size': 8 * 1024 * 1024 * 1024,  # 8GB
                'trt_max_partition_iterations': 1000,
                'trt_min_subgraph_size': 1,
                'trt_fp16_enable': True,  # RTX 5080 Tensor Cores 활용
                'trt_int8_enable': False,  # 필요시 활성화
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': "/mnt/data/exp/exp01/tensorrt_cache",
            }
            providers = [
                ('TensorrtExecutionProvider', tensorrt_options),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
            logger.info("TensorRT EP enabled with RTX 5080 optimizations")
            return providers
        except Exception as e:
            logger.warning(f"TensorRT EP failed: {e}, falling back to CUDA")
    
    # CUDA 폴백
    if "CUDAExecutionProvider" in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # CPU 최종 폴백
    return ['CPUExecutionProvider']
```

**INT8 양자화 캘리브레이션**:
```python
def create_calibration_dataset(data_path: str, num_samples: int = 100):
    """INT8 양자화를 위한 캘리브레이션 데이터셋 생성"""
    import glob
    from PIL import Image
    
    image_paths = glob.glob(f"{data_path}/**/*.jpg", recursive=True)[:num_samples]
    calibration_data = []
    
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        # preprocess와 동일한 전처리 적용
        img_tensor = preprocess([path], img_size=384)[0:1]  # 배치=1
        calibration_data.append(img_tensor.numpy())
    
    return calibration_data

def quantize_model_int8(onnx_model_path: str, calibration_data_path: str):
    """ONNX 모델 INT8 양자화"""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    input_model = onnx_model_path
    output_model = onnx_model_path.replace('.onnx', '_int8.onnx')
    
    calibration_data = create_calibration_dataset(calibration_data_path, 100)
    
    quantize_dynamic(
        input_model,
        output_model,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=['Sigmoid', 'Tanh']  # 정확도 민감 노드 제외
    )
    
    logger.info(f"INT8 quantized model saved: {output_model}")
    return output_model
```

──────────────────────────────────────────────────────────────────────────────
E-9) src/infer.py (공통 전처리·Torch/ONNX/TensorRT 런처)

- **전처리 단일화 강제**: preprocess(paths: list[str], img_size: int) -> Tensor[N,3,H,W]
  - src/data.py의 val 전처리와 **완전 동일** (Resize/CenterCrop/Normalize)
  - **Torch/ONNX가 모두 이 함수를 공유** (중복 구현 금지)
  - 엔진별 정규화/리사이즈/레이아웃 차이로 인한 동등성 문제 방지
- infer_torch(model, batch_tensor, device) -> logits/prob
- infer_onnx(onnx_path, batch_tensor) -> logits/prob
  - select_onnx_providers_tensorrt_first()로 TensorRT 우선 세션 구성
- infer_tensorrt(onnx_path, batch_tensor) -> logits/prob
  - TensorRT 전용 추론 (RTX 5080 최적화)

```
cli: python -m src.infer --engine torch|onnx|tensorrt \
       --model <path> \
       --inputs "/mnt/data/AIHub_576/val/**/*.{jpg,jpeg,png}" \
       --batch 16 \
       --quantize int8  # 선택적 양자화
```

──────────────────────────────────────────────────────────────────────────────
E-9) tests/test_export_compare.py (동등성 테스트 지시)

- 준비: exp_dir/checkpoints/best.pt 가정(없으면 테스트 skip or xfail)
- 시나리오:
  1. export 스크립트 호출(or 함수 직접 호출)로 onnx 생성
  2. val에서 최대 N개 샘플 수집 → preprocess 동일 적용
  3. Torch vs ONNX 실행 → MSE/Top-1 비교
  4. export_report.json 내용/스키마 검증(opset/dynamic_axes/providers/failed=false)
- 경계:
  - GPU 부재 시 CPU 경로로 자동 폴백하는지 확인
  - config.export.input_shape 변경 시 정상 반영되는지 확인
  - onnxruntime 미설치 환경에서는 테스트를 자동 skip(xfail) 처리.

──────────────────────────────────────────────────────────────────────────────
E-10) 실행 예시
$ bash scripts/export_onnx.sh

# 성공 시:

# [OK] Exported: /mnt/data/exp/exp01/export/model-20250810-143012-ab12cd3.onnx

# Report: /mnt/data/exp/exp01/export/export_report.json

# 수동 비교(디버그)

$ python -m src.infer --engine torch --model /mnt/data/exp/exp01/checkpoints/best.pt --inputs "/mnt/data/AIHub_576/val/**/\*.jpg" --batch 16
$ python -m src.infer --engine onnx --model /mnt/data/exp/exp01/export/model-20250810-143012-ab12cd3.onnx --inputs "/mnt/data/AIHub_576/val/**/\*.jpg" --batch 16

──────────────────────────────────────────────────────────────────────────────
E-11) 트러블슈팅

- Unsupported operator:
  → opset 17 유지 권장. 안되면 임시로 18 시도 후 ORT 버전 업그레이드 검토.
- 값 불일치(MSE↑/Top-1 mismatch):
  → fp16 환경에서는 완화된 기준 적용(MSE ≤ 5e-4, Top-1 mismatch ≤ 2%)
  → 전처리 불일치(정규화/크롭) 확인, 모델 eval/float 재확인, dynamic_axes 누락 점검
- CUDA EP 실패:
  → nvidia-smi/드라이버 확인, onnxruntime-gpu 설치 고려(기본은 onnxruntime). 자동 CPU 폴백 동작 확인.
- 메모리 부족:
  → 배치 축소, 224→192/160 임시 축소, do_constant_folding=True 유지.

## 🎯 **PART_E 핵심 업데이트 완료**

### ✅ **사용자 제어 Two-Stage ONNX Export**
- **단일 약품**: EfficientNetV2-S → ONNX (384px 입력)
- **조합 약품**: YOLOv11m → ONNX (640px 입력)  
- **Dynamic Axes**: 배치 크기 가변 지원

### ✅ **RTX 5080 최적화 Export**
- **ONNX Runtime**: GPU EP 자동 선택, CUDA/CPU 폴백
- **모델 최적화**: graph optimization, memory pattern 최적화
- **양자화**: INT8 동적 양자화 옵션 (정확도 허용 범위 내)

### ✅ **실용적 검증 & 품질 보장**
- **PyTorch vs ONNX**: MSE ≤ 1e-4, Top-1 mismatch ≤ 0.1%
- **성능 비교**: 추론 속도, 메모리 사용량 비교
- **fp16 완화**: fp16 내보내기 시 완화된 기준 적용

**✅ PART_E 완룈: 사용자 제어 Two-Stage ONNX Export 및 실용적 검증**
