# Part D — 조건부 Two-Stage 학습 + RTX 5080 16GB 최적화

**목적**: RTX 5080 16GB + 128GB RAM 최적화 학습, 조건부 Two-Stage (단일→직접분류, 조합→검출후분류), 안정성(OOM 폴백), 가시성(TensorBoard), 재현성(체크포인트)

[전제/고정]

- **경로**: /home/max16/pillsnap (코드), **/home/max16/pillsnap_data** (Native Linux SSD, 프로젝트 분리), **/home/max16/pillsnap_data/exp/exp01** (실험, Native Linux SSD)
- **하드웨어**: AMD Ryzen 7800X3D + 128GB RAM + RTX 5080 16GB
- **RAM 활용**: 라벨 캐시, 배치 프리페치, 이미지 메타데이터 캐시, 워커 메모리 공유
- **핵심**: 사용자 제어 Two-Stage (pipeline_strategy: "user_controlled")
- **모델**: YOLOv11m (검출, 640px) + EfficientNetV2-S (분류, 384px, 5000클래스)
- **모델 파일**: `detector_yolo11m.py`, `classifier_efficientnetv2_s.py`
- **최적화**: AMP auto, TF32, channels_last, torch.compile, 배치크기 자동조정

[이 파트에서 구현/수정할 파일] (현재 구조 반영)

1. src/utils/core.py  # ✅ 구현 완료 (ConfigLoader, PillSnapLogger)

   - load_config(cfg_path: str) -> DictConfig|dict
   - set_seed(seed: int, deterministic: bool) -> None
   - build_logger(exp_dir: str, name="pillsnap") -> logging.Logger # 콘솔+파일 핸들러
   - StepTimer # data/compute 타이밍 헬퍼
   - TBWriter(exp_dir: str) # tensorboard 세션(없으면 안전 폴백)
   - save_json(path: str, obj: dict) / read_json(path: str) -> dict
   - ensure_dir(path: str) -> None
   - get_git_sha() -> str|"nogit"
   - select_amp_dtype(policy: "auto"|"fp16"|"bf16") -> torch.dtype
   - enable_tf32(flag: bool) -> None
   - try_torch_compile(model, mode: str) -> nn.Module
   - cuda_mem_mb() -> float
   - auto_find_batch_size(fn_train_step, init_bs: int, min_bs=1, max_bs=None, amp_force_fp16=False) -> int # Annex1
   - autotune_num_workers(build_loader_fn, candidates=[4,8,12,16], warmup=30, steps=100) -> int # Annex2
   - save_ckpt(path, obj) / load_ckpt(path) -> dict # Annex8

2. src/train.py

   - main(args|cfg): Two-Stage 학습 엔트리 (classification 우선, detection은 필요시) + Stage 4 완료시 final_test_evaluation 호출
   - train_classification_stage(cfg, logger) -> classification_model_path  # 단일 약품용 (주력)
   - train_detection_stage(cfg, logger) -> detection_model_path  # 조합 약품용 (선택적)
   - build_classification_model(backbone: str, num_classes: int, channels_last: bool) -> nn.Module
   - build_yolo_model(model_size: str, num_classes: int, lazy_load: bool=True) -> YOLO model | None
   - build_criterion_detection() -> YOLOv11 내장 loss
   - build_criterion_classification(cfg, class_weights: Tensor|None) -> nn.Module
   - build_optimizer(cfg, params) -> torch.optim.Optimizer
   - build_scheduler(cfg, optimizer, steps_per_epoch: int) -> torch.optim.lr_scheduler._LRScheduler
   - train_one_epoch_classification(model, loader, optimizer, scaler, criterion, scheduler, cfg, logger, tb, epoch_idx) -> dict
   - train_one_epoch_detection(yolo_model, train_loader, cfg, logger, tb, epoch_idx) -> dict  # 조합용
   - validate_classification(model, loader, criterion, cfg, logger, tb, epoch_idx) -> dict
   - validate_detection(yolo_model, val_loader, cfg, logger, tb, epoch_idx) -> dict # mAP 계산
   - evaluate_classification_metrics(y_true, y_pred, y_prob, topk=[1,5]) -> dict # accuracy, macro_f1, top5, confidence
   - final_test_evaluation(cfg, test_loaders, model_paths, logger, exp_dir) -> dict  # Stage 4 완료 후 test 최종 평가
   - evaluate_detection_metrics(predictions, targets) -> dict # mAP@0.5, mAP@0.5:0.95
   - predict_pipeline(image, mode="single", conf_threshold=0.3) -> dict  # 통합 추론 파이프라인
   - early_stopping(state, metric_name, mode="max", patience=7, min_delta=0.0) -> (stop: bool, best:bool)
   - oom_recovery_state_machine(...) -> Updated cfg knobs / signals # Annex6
   - (옵션) EMA: if cfg.train.ema.enabled: update/validate with EMA weights # Annex5

3. src/evaluate.py

   - plot_confusion_matrix(cm, class_names, out_path)
   - plot_detection_results(images, predictions, out_dir) # bbox 시각화
   - plot_roc_pr_curves(y_true_onehot, y_prob, out_dir) # 분류용 저장
   - dump_metrics_json(metrics: dict, out_path) # {detection: {mAP@0.5, mAP@0.5:0.95}, classification: {acc,macro_f1,top5,loss}}

4. scripts/train.sh (Part B 골격을 실제 로직으로 채움)
   - venv 활성화 → config 경로 확인 → exp_dir/{logs,tb,reports,checkpoints,export} 보장
   - 핵심 구성 echo 후 python -m src.train --cfg config.yaml 실행
   - stdout/stderr를 exp_dir/logs/train.out|err로 리다이렉트하고 종료코드별 메시지

[Interleaved Two-Stage 학습 파이프라인 상세 규칙]

D-1) Interleaved Learning Strategy

- 전략: "interleaved" - 미니배치 단위로 검출/분류 교차 학습
- 비율: [1, 1] - det 1스텝 → cls 1스텝 교대 반복
- 구현: 
  ```python
  d, c = cfg.train.interleave_ratio  # [1, 1]
  for step in range(total_steps):
      if step % (d + c) < d:
          train_detection_step()
      else:
          train_classification_step()
  ```
- 효과: GPU 유휴시간 50% → 10% 감소, 전체 학습시간 30% 단축

D-2) 디바이스/정밀도/포맷 (Both Stages)

- device = "cuda" if torch.cuda.is_available() else "cpu"
- TF32: cfg.train.detection/classification.tf32 → enable_tf32(True) ⇒
  torch.backends.cuda.matmul.allow_tf32=True, torch.backends.cudnn.allow_tf32=True,
  torch.set_float32_matmul_precision("high")
- AMP:
  - Detection: YOLOv11 자체 AMP 사용
  - Classification: dtype = select_amp_dtype(cfg.train.classification.amp_dtype) # "auto"면 bf16 지원 시 bfloat16, 아니면 fp16
    autocast(enabled=cfg.train.classification.amp, dtype=dtype) + GradScaler(enabled=(cfg.train.classification.amp and dtype==torch.float16))
  - 주의: dtype==bfloat16 인 경우 GradScaler는 불필요하므로 비활성.
- channels_last: 분류 모델에만 적용
  if cfg.train.classification.channels_last=True: model = model.to(memory_format=torch.channels_last); 입력 텐서도 to(memory_format=channels_last)로 전송
- torch.compile:
  - MVP: "reduce-overhead" (안정성 우선)
  - Advanced+: "max-autotune" (RTX 5080 공격적 최적화)
  - 워밍업: 100 스텝 컴파일 안정화
  - 실패 시 폴백 모드로 자동 전환 후 로그

D-3) 모드별 데이터로더 구성(Part C 연동)

- from src.data import build_dataloaders_twostage
- 반환: cls_train_loader, cls_val_loader, cls_test_loader|None, det_train_loader(optional), det_val_loader(optional), det_test_loader(optional), meta
- Classification (주력): 단일 약품 직접 분류용 데이터로더
- Detection (즉시 로드): 조합 약품용 YOLO 포맷 데이터로더 (128GB RAM으로 즉시 로드)
- 중요: test_loader들은 Stage 4 완료 전에는 None 반환, Stage 4 완료 후에만 생성하여 최종 평가시 사용
- cfg.dataloader.autotune_workers=True면 autotune_num_workers()로 후보 [4,8,12,16] 중 선택 → 최종 로더 재생성
- class_weights: cfg.loss.use_class_weights=True면 분류 단계에서만 compute_class_weights(...) 적용
- 로더 공통: pin_memory, persistent_workers, prefetch_factor, drop_last, safe_collate
  - PyTorch≥2.0이면 pin_memory_device="cuda" 설정을 고려(입출력 고정 시 효율 ↑).
- 로깅 강화(게이팅/혼동 케이스): step/epoch 로그와 TB에 다음을 추가 기록한다.
  - gating_score, mode_used, single_confidence, fallback_reason("low_confidence"|"multi_object_hint"|"manual_combo")
  - hard_cases 저장 조건(예): (mode="single" & single_confidence<0.30) or (mode="auto" & mode_used="combo") or misroute 추정

D-2.1) 128GB RAM 최적화 전략

- **메모리 할당 계획 (현실적 기본선)**:
  - OS + 기본: ~8GB
  - 핫셋 캐시: ~25GB (6만장 × 384px)
  - 배치 프리페치: ~8GB (prefetch_factor=4 × 2GB/배치)
  - 라벨 캐시: ~2GB (메타데이터)
  - 워커 버퍼: ~8GB (8 워커 × 1GB)
  - 여유 메모리: ~77GB (확장 여유분)

- **구체적 구현 (기본선)**:
  - `cache_policy="hotset"`: 핫셋 6만장만 캐시 (Phase 1 기본)
  - `use_lmdb=false`: 기본 비활성, data_time 병목 시에만 활성화
  - `preload_samples=0`: 기본 미사용, 필요시 단계적 증가
  - `prefetch_factor=4`: 현실적 수준 (8→4로 감소)
  - `num_workers=8`: 현실적 워커 수 (16→8로 감소)

D-3) 모델/손실/옵티마/스케줄러

- 분류 모델 (주력): timm.create_model(cfg.model.backbone, pretrained=True, num_classes=cfg.model.num_classes)
  - channels_last면 모델 메모리 포맷 변환
- 검출 모델 (선택적): YOLO 지연 로딩 - cfg.inference.lazy_load_detector=True면 필요시만 로드
- 손실: CrossEntropy(label_smoothing=cfg.loss.label_smoothing, weight=class_weights|None)
- 옵티마: AdamW(lr=cfg.train.lr, weight_decay=cfg.train.weight_decay, fused=True) # torch≥2.0 & CUDA일 때만; 미지원 시 자동 False
- 스케줄러: cfg.train.scheduler == "cosine" → CosineAnnealing + warmup_steps
  (onecycle 옵션은 후속 확장 가능)
- zero_grad(set_to_none=cfg.train.zero_grad_set_to_none)

D-4) 동적 배치 튜닝 (Auto Batch, Annex1)

- cfg.train.detection.auto_batch_tune=True/classification.auto_batch_tune=True이면
  1. max_batch = cfg.train.detection.auto_batch_max (16) / classification.auto_batch_max (64)
  2. vram_headroom = 0.88 (12% 여유로 torch.compile 오버헤드 대응)
  3. fn_train_step는 몇 step만 실제 forward/backward까지 수행
  4. OOM 발생 시: torch.cuda.empty_cache() → grad_accum 증가 → batch 반감 순
  5. 확정된 batch_size로 train_loader 재생성, 로그/TB 기록

D-5) 학습 루프(1 epoch + Interleaved Strategy)

**Interleaved Two-Stage 학습 (PART_B 설정 반영)**:
```python
# cfg.train.strategy == "interleaved"인 경우
d, c = cfg.train.interleave_ratio  # [1, 1]
det_loader_iter = iter(det_loader)
cls_loader_iter = iter(cls_loader)

for global_step in range(total_steps):
    if global_step % (d + c) < d:
        # Detection 학습 스텝
        batch = next(det_loader_iter)
        loss = train_detection_step(det_model, batch, ...)
    else:
        # Classification 학습 스텝
        batch = next(cls_loader_iter)
        loss = train_classification_step(cls_model, batch, ...)
```

**기본 학습 루프 (각 모델별)**:
- Timer로 step별 data_time/compute_time 측정
- for step, (x,y,meta) in enumerate(train_loader):
  - x,y를 device로 전송(to(..., non_blocking=True)); channels_last면 x = x.to(memory_format=torch.channels_last)
  - with autocast(...):
    logits = model(x)
    loss = criterion(logits, y) / cfg.train.grad_accum_steps
  - scaler.scale(loss).backward()
  ```python
  if (step + 1) % grad_accum_steps == 0:
    if cfg.train.grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
  ```
  - compute throughput = (`effective_batch = batch_size * grad_accum_steps`) / (data_time + compute_time)
  - step_log_interval마다:
    logger.info(f"[{epoch}/{step}] loss=... lr=... data=... compute=... thr=... cuda={cuda_mem_mb():.1f}")
    TB: train/loss, train/lr, perf/data_time, perf/compute_time, perf/throughput, perf/cuda_mem
- (옵션) EMA: step마다 ema.update(model)

D-6) 검증/지표/얼리스톱

- validate():
  - Stage별 학습 중에는 val_loader만 사용 (test_loader 절대 사용 금지)
  - no_grad + autocast(enabled=False) # 안정성 위해 검증은 fp32
  - 전체 로더 순회하여 loss/정확도/topk 수집; macro_f1 계산(sklearn)
  - y_prob = softmax(logits); y_pred = argmax
- 평가 지표(dict):
  { "val_loss":..., "acc":..., "macro_f1":..., "top5":..., "n":... }
  → TB에 val/accuracy, val/macro_f1, val/loss 로 기록
- 검출 mAP 계산은 Torch/ONNX 공통 NMS를 적용한 결과 기준으로 수행한다.
- early_stopping():
  - monitor=cfg.train.early_stopping.monitor (분류: macro_f1, 검출: mAP@0.5), mode="max"
  - patience, min_delta 적용
  - best 갱신 시 best.pt 저장(Annex8 스키마)
  - stop=True면 학습 종료

# 최종 평가 (Stage 4 완료 후)
- final_test_evaluation(cfg, test_loaders, model_paths, logger, exp_dir):
  • 전제조건: Stage 4 (500K, 5000클래스) 완료 및 best.pt 존재 확인
  • 모델 로딩: classification_best.pt, detection_best.pt (있는 경우)
  • Test 데이터 평가: test_loaders (단일+조합) 1회 실행, 학습중 절대 사용 금지 확인
  • 성능 비교: val_metrics vs test_metrics 차이 계산 (overfitting 진단)
  • 목표 달성 확인: 단일 92% (macro_f1), 조합 mAP@0.5=0.85
  • 결과 저장: exp_dir/reports/final_test_results.json
  • 상용화 권고: 목표 달성시 "READY_FOR_PRODUCTION", 미달시 개선사항 제시
  • 반환 구조: {"test_results": {"classification": {"accuracy": 0.921, "macro_f1": 0.889}, "detection": {"mAP_0.5": 0.847}}, "business_goals": {"overall_target_met": false}, "recommendation": "NEEDS_IMPROVEMENT", "ready_for_production": false, "improvements": [...]}  

D-7) 체크포인트/재개(Annex8)

- last.pt: 매 epoch 저장(모델/opt/sched/scaler/epoch/best_metric/config 포함)
- best.pt: monitor 기준 최고일 때만 갱신
- resume:
  - cfg.train.resume == "last"면 자동 탐색 로드(없으면 경고)
  - 실수 방지: 모델 백본/num_classes 불일치 시 친절한 에러로 중단

D-8) OOM 폴백 상태 머신 + 메모리 모니터링(Annex6)

- 메모리 모니터링(예방):
  - 매 epoch 시작 시 GPU 메모리 사용률 체크
  - 80% 초과 시 경고 로그(oom/memory_warning)
  - 사용률과 배치 크기 상관관계 TB 기록
- CUDA OOM RuntimeError 감지 시 가드레일 적용:
  ```python
  def oom_recovery_state_machine(retries, current_state, cfg):
      \"\"\"OOM 복구를 위한 학습 일관성 보장 상태 머신\"\"\"
      retries += 1
      if retries > cfg.train.oom.max_retries:  # 기본 4회
          return {"action": "emergency_exit", "save_checkpoint": True}
      
      time.sleep(cfg.train.oom.cooldown_sec)  # 기본 2초
      
      # 현재 글로벌 배치 크기
      G_old = current_state["batch_size"] * current_state["grad_accum"]
      
      # S1: AMP fp16 강제 (1회만, 모델/수학 그대로)
      if cfg.train.oom.escalate_to_fp16 and not current_state.get("did_amp_escalate"):
          logger.warning(f"OOM Recovery S1: Escalating to fp16 (preserving schedules)")
          return {
              "action": "amp_fp16",
              "preserve_schedules": True,  # 스케줄링 변경 없음
              "next_state": {**current_state, "did_amp_escalate": True}
          }
      
      # S2: 마이크로배칭 (글로벌 배치 유지 우선)
      if current_state["grad_accum"] < cfg.train.oom.max_grad_accum:
          bs_new = max(current_state["batch_size"] // 2, cfg.train.oom.min_batch)
          accum_new = min(math.ceil(G_old / bs_new), cfg.train.oom.max_grad_accum)
          G_new = bs_new * accum_new
          
          if G_new == G_old:  # 글로벌 배치 유지됨
              logger.warning(f"OOM Recovery S2: Microbatching bs={bs_new}, accum={accum_new} (global batch preserved: {G_old})")
              return {
                  "action": "microbatching",
                  "batch_size": bs_new,
                  "grad_accum": accum_new,
                  "preserve_schedules": True,  # LR/WD/EMA 변경 없음
                  "replay_batch": True
              }
      
      # S3: 글로벌 배치 변경 (불가피할 때만)
      bs_new = max(current_state["batch_size"] // 2, cfg.train.oom.min_batch)
      accum_new = min(cfg.train.oom.max_grad_accum, math.ceil(G_old / bs_new))
      G_new = bs_new * accum_new
      
      # 학습 일관성 보장: 샘플 기준 스케줄링
      lr_scale = G_new / G_old  # Linear Scaling Rule
      steps_per_epoch_ratio = G_old / G_new
      
      logger.warning(f"OOM Recovery S3: Global batch change G_old={G_old}→G_new={G_new}, lr_scale={lr_scale:.4f}, by_samples scheduling")
      return {
          "action": "global_batch_change",
          "batch_size": bs_new,
          "grad_accum": accum_new,
          "lr_scale": lr_scale,
          "wd_scale": steps_per_epoch_ratio,  # 에폭당 WD 총량 유지
          "scheduler_mode": "by_samples",     # 샘플 기준 전환
          "replay_batch": True,               # 동일 시드로 재실행
          "audit_log": {"G_old": G_old, "G_new": G_new, "lr_scale": lr_scale}
      }
      
      # S3: grad_accum 증가 (상한 체크)
      if grad_accum < cfg.train.oom.max_grad_accum:  # 기본 8
          grad_accum = min(grad_accum * 2, cfg.train.oom.max_grad_accum)
          return retries, did_empty_cache, did_amp_escalate, grad_accum, batch_size
      
      # S4: batch_size 감소 (하한 체크)
      if batch_size > cfg.train.oom.min_batch:  # 기본 1
          batch_size = max(batch_size // 2, cfg.train.oom.min_batch)
          rebuild_dataloader(batch_size)
          return retries, did_empty_cache, did_amp_escalate, grad_accum, batch_size
      
      fail("min_batch & max_grad_accum reached")
  ```
- **OOM 폴백 핵심 원칙**:
  - **상한 고정**: max_retries=4, max_grad_accum=8, min_batch=1
  - **우선순위**: 글로벌 배치 보존 → 선형 스케일링 → 최후 배치 축소
  - **AMP 정책**: auto(bf16>fp16) 기본, OOM 시 1회 fp16 강제만
  - **로깅 강화**: 각 단계 진입/성공/실패를 상세 로그+TB(oom/retries)

- **config.yaml 필수 OOM 가드레일 키** (없으면 실패):
  ```yaml
  train:
    oom:
      max_retries: 4              # 최대 재시도 횟수
      max_grad_accum: 8           # 최대 gradient accumulation
      min_batch: 1                # 최소 배치 크기
      cooldown_sec: 2             # 재시도 간 대기 시간
      escalate_to_fp16: true      # AMP fp16 강제 전환 허용
  ```

D-9) 리포트/시각화

- src/evaluate.py:
  - 혼동행렬(cm) 저장, ROC/PR 커브(다중 클래스면 macro/one-vs-rest) 저장
  - reports/metrics.json에 epoch-level 누적/최신 저장(save_json)
- scripts/train.sh:
  - 종료 후 "SUCCESS"면 {exp_dir}/reports/metrics.json의 최신 지표 파일 경로를 echo
  - hard_cases/*.jsonl 및 입력 썸네일/크롭(옵션)을 보존하여 주간 분석에 활용

D-10) 로깅/프로파일

- 로그 라인 공통 포맷:
  [train] e={epoch}/{max} s={step}/{len} loss={:.4f} lr={:.2e} data={:.3f}s comp={:.3f}s thr={:.1f}/s cuda={:.0f}MB
- profile_interval(기본 50 step)마다 상세 라인 덤프
- tensorboard 디렉터리: {exp_dir}/tb (TBWriter가 자동 생성)
  - (설치 필요) pip install tensorboard

D-11) 재현성/성능 모드

- set_seed(cfg.train.seed, cfg.train.deterministic)
- torch.use_deterministic_algorithms(cfg.train.deterministic, warn_only=True)
- deterministic=False(성능 모드)일 때 cudnn.benchmark=True
- True일 때 cudnn.deterministic=True & benchmark=False

D-12) 단순 Go/No-Go 평가 시스템

```python
STAGE_TARGETS = {
  1: {"min_accuracy": 0.78, "max_latency_ms": 220, "min_coverage": 0.20,
      "min_eval_samples": 2000, "max_acc_drop": 0.02, "epsilon_acc": 0.002},
  2: {"min_accuracy": 0.82, "max_latency_ms": 220, "min_coverage": 0.40,
      "min_eval_samples": 5000, "max_acc_drop": 0.02, "epsilon_acc": 0.002},
  3: {"min_accuracy": 0.85, "max_latency_ms": 200, "min_coverage": 0.70,
      "min_eval_samples": 10000, "max_acc_drop": 0.01, "epsilon_acc": 0.002},
  4: {"min_accuracy": 0.85, "max_latency_ms": 200, "min_coverage": 0.70,
      "min_eval_samples": 10000, "max_acc_drop": 0.01, "epsilon_acc": 0.002}
}

def simple_stage_evaluation(stage: int, results: dict, baseline: dict = None) -> tuple[bool, list[str]]:
    """Go/No-Go 평가 with 5개 확장 포인트"""
    t = STAGE_TARGETS[stage]
    reasons = []
    
    # 1. 지연시간 기준 (p95)
    p95 = results.get("latency_p95_ms", float("inf"))
    ok_lat = p95 <= t.get("max_latency_ms", float("inf"))
    if not ok_lat: reasons.append("LATENCY_P95_TOO_HIGH")
    
    # 2. 표본/커버리지
    n = int(results.get("n_eval_samples", 0))
    ok_n = n >= t.get("min_eval_samples", 0)
    if not ok_n: reasons.append("NOT_ENOUGH_SAMPLES")
    
    cov = results.get("class_coverage", 0.0)
    ok_cov = cov >= t.get("min_coverage", 0.0)
    if not ok_cov: reasons.append("COVERAGE_TOO_LOW")
    
    # 3. 리그레션 가드
    acc = results.get("accuracy", 0.0)
    ok_reg = True
    if baseline and "accuracy" in baseline:
        max_drop = t.get("max_acc_drop", 0.0)
        if acc < baseline["accuracy"] - max_drop:
            ok_reg = False
            reasons.append("REGRESSION_TOO_LARGE")
    
    # 4. 히스테리시스
    epsilon = t.get("epsilon_acc", 0.0)
    ok_acc = acc >= (t["min_accuracy"] - epsilon)
    if not ok_acc: reasons.append("ACC_BELOW_TARGET")
    
    # 5. 사유 반환
    ok_crash = bool(results.get("no_crashes", False))
    if not ok_crash: reasons.append("CRASH_DETECTED")
    
    ok_mem = bool(results.get("memory_stable", False))
    if not ok_mem: reasons.append("MEMORY_UNSTABLE")
    
    go = all([ok_lat, ok_n, ok_cov, ok_reg, ok_acc, ok_crash, ok_mem])
    return go, reasons
```

D-13) Stage별 하드웨어 설정 오버라이드

**Stage별 하드웨어 설정 오버라이드 로직**:
```python
# src/utils.py - apply_stage_overrides 함수
def apply_stage_overrides(cfg, current_stage: int) -> dict:
    """Stage별 하드웨어 설정을 동적으로 적용"""
    stage_key = f"stage_{current_stage}"
    stage_overrides = cfg.train.get("stage_overrides", {}).get(stage_key, {})
    
    if not stage_overrides:
        logger.info(f"Stage {current_stage}: No overrides configured")
        return cfg
    
    original_cfg = deepcopy(cfg)
    applied_changes = []
    
    # 배치 크기 오버라이드
    if "batch_size_override" in stage_overrides:
        override_bs = stage_overrides["batch_size_override"]
        cfg.train.batch_size = override_bs
        applied_changes.append(f"batch_size: {override_bs}")
    
    # 컴파일 모드 오버라이드
    if "compile_mode" in stage_overrides:
        override_mode = stage_overrides["compile_mode"]
        cfg.optimization.torch_compile = override_mode
        applied_changes.append(f"torch_compile: {override_mode}")
    
    # 메모리 포맷 오버라이드
    if "memory_format" in stage_overrides:
        override_format = stage_overrides["memory_format"]
        cfg.optimization.channels_last = (override_format == "channels_last")
        applied_changes.append(f"channels_last: {override_format == 'channels_last'}")
    
    # 워커 수 오버라이드
    if "num_workers_override" in stage_overrides:
        override_workers = stage_overrides["num_workers_override"]
        cfg.dataloader.num_workers = override_workers
        applied_changes.append(f"num_workers: {override_workers}")
    
    logger.info(f"Stage {current_stage} overrides applied: {', '.join(applied_changes)}")
    return cfg
```

**Stage별 최적화 전략**:
```python
# src/train.py - main 함수 내 Stage 오버라이드 적용
def main(cfg):
    current_stage = cfg.data.progressive_validation.current_stage
    
    # Stage별 하드웨어 설정 적용
    cfg = apply_stage_overrides(cfg, current_stage)
    
    # Stage별 특화 최적화
    if current_stage <= 2:  # 작은 데이터셋
        # 작은 배치, 빠른 컴파일, 적은 워커
        enable_small_dataset_optimizations(cfg)
    elif current_stage >= 3:  # 대용량 데이터셋
        # 큰 배치, 최대 최적화, 많은 워커
        enable_large_dataset_optimizations(cfg)
    
    logger.info(f"Stage {current_stage} hardware optimization enabled")

def enable_small_dataset_optimizations(cfg):
    """Stage 1-2: 작은 데이터셋 최적화"""
    # 빠른 프로토타이핑을 위한 설정
    if cfg.train.detection.auto_batch_max > 64:
        cfg.train.detection.auto_batch_max = 64  # 작은 배치
    if cfg.train.classification.auto_batch_max > 96:
        cfg.train.classification.auto_batch_max = 96
    
    # 컴파일 오버헤드 최소화
    if cfg.optimization.torch_compile == "max-autotune":
        cfg.optimization.torch_compile = "reduce-overhead"
    
    logger.info("Small dataset optimizations applied")

def enable_large_dataset_optimizations(cfg):
    """Stage 3-4: 대용량 데이터셋 최적화"""
    # 최대 성능을 위한 설정
    cfg.optimization.torch_compile = "max-autotune"
    cfg.optimization.channels_last = True
    
    # VRAM 최대 활용
    if cfg.train.auto_batch_vram_headroom < 0.90:
        cfg.train.auto_batch_vram_headroom = 0.92
    
    logger.info("Large dataset optimizations applied")
```

D-14) Stage별 차별화된 평가 기준 (실제 데이터 기반)

[Stage 1: 파이프라인 검증 (5,000장, 50클래스)]
```yaml
mandatory_checks:
  - pipeline_complete: true           # 전체 파이프라인 에러 없이 완료
  - detection_model_loading: true     # YOLOv11x 모델 정상 로딩
  - classification_model_loading: true # EfficientNetV2-L 모델 정상 로딩
  - onnx_conversion_success: true     # 두 모델 모두 ONNX 변환 성공

performance_targets:
  detection:
    mAP_0.5: ≥0.30                  # 기본 검출 가능성 확인
    inference_time_ms: ≤50          # RTX 5080에서 640px 실시간 처리 가능성
  classification:
    accuracy: ≥0.40                 # 50클래스 기준 (무작위 2% × 20배)
    inference_time_ms: ≤10          # 384px 분류 실시간 가능
  system:
    gpu_memory_gb: ≤14              # VRAM 안정성 확인
    data_loading_s: ≤2              # 128GB RAM 활용도 검증

recommendation_thresholds:
  RECOMMEND_PROCEED: all_mandatory + all_performance_targets
  SUGGEST_OPTIMIZE: performance >= 70% of targets
  WARN_STOP: performance < 70% of targets
```

[Stage 2: 기본 성능 (25,000장, 250클래스)]
```yaml
mandatory_checks:
  - auto_batch_tuning_success: true   # Auto Batch 튜닝 성공
  - tensorboard_logging: true         # TensorBoard 로깅 정상
  - class_balance_verification: true  # 클래스 균형 검증 통과
  - memory_optimization_working: true # 128GB RAM 최적화 동작

performance_targets:
  detection:
    mAP_0.5: ≥0.50                  # 기본 검출 성능
    mAP_0.5_0.95: ≥0.35             # COCO 표준 지표
    precision: ≥0.45                # 검출 정밀도
    recall: ≥0.40                   # 검출 재현율
  classification:
    accuracy: ≥0.60                 # 250클래스 대상 향상된 성능
    macro_f1: ≥0.55                 # 클래스 불균형 환경에서의 실제 성능
    top5_accuracy: ≥0.80            # 상위 5개 후보 정확도
  optimization:
    batch_size_achieved: ≥8         # RTX 5080 메모리 효율성
    throughput_img_s: ≥100          # 처리량 기준

advanced_metrics:
  - class_imbalance_ratio: ≤10:1    # 클래스 불균형 관리
  - convergence_stability: loss_variance < 0.1
  - memory_efficiency: peak_usage/avg_usage < 1.5
```

[Stage 3: 확장성 테스트 (100,000장, 1,000클래스)]
```yaml
mandatory_checks:
  - scalability_test_pass: true      # 확장성 테스트 통과
  - multi_batch_processing: true     # 다중 배치 처리 가능
  - memory_leak_test: true           # 메모리 누수 없음
  - stability_test_pass: true        # 장시간 안정성 테스트

performance_targets:
  detection:
    mAP_0.5: ≥0.70                  # 높은 검출 성능
    mAP_0.5_0.95: ≥0.50             # 고급 검출 성능
    recall_0.5: ≥0.75               # 검출율
    precision_0.5: ≥0.70            # 검출 정밀도
  classification:
    accuracy: ≥0.75                 # 1,000클래스 대상 높은 성능
    macro_f1: ≥0.70                 # 불균형 환경에서 높은 성능
    top5_accuracy: ≥0.90            # 높은 후보 정확도
    per_class_min_f1: ≥0.30         # 최소 클래스 성능 보장
  scalability:
    multi_batch_size: ≥32           # 배치 처리 효율성
    memory_stable_hours: ≥12        # 장시간 안정성
    api_response_time: ≤100ms       # API 응답 시간
```

[Stage 4: 최종 프로덕션 (500,000장, 5,000클래스)]
```yaml
mandatory_checks:
  - full_dataset_processing: true    # 전체 데이터셋 처리 완료
  - production_api_deployment: true  # 프로덕션 API 배포 성공
  - cloudflare_tunnel_setup: true    # Cloudflare Tunnel 설정 완료
  - automated_monitoring: true       # 자동 모니터링 시스템 동작

performance_targets:
  detection:
    mAP_0.5: ≥0.85                  # PART_A 최종 목표
    mAP_0.5_0.95: ≥0.65             # 최고 수준 검출 성능
    recall_0.5: ≥0.90               # 최고 수준 검출율
    precision_0.5: ≥0.85            # 최고 수준 정밀도
  classification:
    accuracy: ≥0.92                 # PART_A 최종 목표
    macro_f1: ≥0.88                 # 조기종료 지표 통일
    top5_accuracy: ≥0.98            # 거의 완벽한 top-5 성능
    top1_per_class: ≥0.85           # 모든 클래스에서 높은 성능
  production:
    inference_latency_ms: ≤30       # 전체 파이프라인 실시간 성능
    throughput_img_s: ≥200          # 높은 처리량
    api_uptime: ≥99.9%              # 높은 가용성
```

D-14) OptimizationAdvisor 권장 시스템 (구체적 구현)

```python
class OptimizationAdvisor:
    """학습 실패 시 구체적 조정 제안 시스템"""
    
    def evaluate_and_recommend(self, stage: int, results: dict) -> dict:
        """성능 평가 후 권장사항 생성 → 사용자 결정"""
        
        # 공통 중단 기준 (모든 Stage)
        if (results.get("repeated_oom", 0) > 5 or
            math.isnan(results.get("loss", 0)) or
            results.get("data_corruption_rate", 0) > 0.01):
            return self._critical_failure_recommendation()
        
        # Stage별 차별화된 평가
        stage_configs = get_stage_config(stage)
        
        # 필수 체크 확인
        mandatory_pass = all(
            results["mandatory_checks"].get(check, False) 
            for check in stage_configs["mandatory_checks"]
        )
        
        if not mandatory_pass:
            return self._mandatory_failure_recommendation(stage, results, stage_configs)
        
        # 성능 기준 확인 및 권장사항 생성
        performance_score = calculate_performance_score(results, stage_configs)
        suggestions = self.diagnose_and_suggest(results, stage)
        
        # 권장사항 생성 (자동 결정 X, 사용자 선택권 제공)
        if performance_score >= 1.0:  # 모든 목표 달성
            recommendation = {
                "recommendation": "RECOMMEND_PROCEED",
                "color": "🟢", 
                "reason": "all_targets_met",
                "message": f"Stage {stage} 모든 목표 달성!",
                "performance_score": performance_score,
                "user_options": [
                    "[1] Stage {stage+1}로 진행",
                    "[2] 현재 Stage에서 추가 최적화", 
                    "[3] 상세 분석 리포트 생성"
                ]
            }
        elif performance_score >= 0.7:  # 70% 이상 달성
            recommendation = {
                "recommendation": "RECOMMEND_PROCEED",
                "color": "🟢",
                "reason": "sufficient_performance", 
                "message": f"Stage {stage} 충분한 성능 달성",
                "performance_score": performance_score,
                "user_options": [
                    "[1] 현재 성능으로 다음 Stage 진행",
                    "[2] 권장 최적화 적용 후 재시도",
                    "[3] 수동 하이퍼파라미터 조정"
                ]
            }
        elif performance_score >= 0.5:  # 50% 이상 달성
            recommendation = {
                "recommendation": "SUGGEST_OPTIMIZE",
                "color": "🟡",
                "reason": "performance_below_target",
                "message": f"Stage {stage} 성능 미달. 최적화 권장",
                "suggestions": suggestions,
                "user_options": [
                    "[1] 권장사항 적용 후 재시도",
                    "[2] 현재 성능으로 진행 (위험)",
                    "[3] 상세 디버깅 모드"
                ]
            }
        else:  # 50% 미만
            recommendation = {
                "recommendation": "WARN_STOP",
                "color": "🔴", 
                "reason": "performance_too_low",
                "message": f"Stage {stage} 성능이 매우 낮음",
                "performance_score": performance_score,
                "suggestions": suggestions,
                "user_options": [
                    "[1] 강력한 최적화 적용 후 재시도",
                    "[2] 아키텍처 재검토",
                    "[3] 데이터 품질 점검"
                ]
            }
        
        # 사용자에게 선택권 제공
        return self._present_recommendation_to_user(recommendation)
    
    def _present_recommendation_to_user(self, recommendation: dict) -> dict:
        """터미널에 권장사항 출력 및 사용자 선택 대기"""
        
        print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    🎯 Stage 평가 완료                             ║
╠══════════════════════════════════════════════════════════════════╣
║ {recommendation['color']} {recommendation['message']}
║ 
║ 📊 성능 점수: {recommendation.get('performance_score', 'N/A'):.3f}
║ 
║ 💡 권장사항:
""")
        
        if 'suggestions' in recommendation:
            for i, suggestion in enumerate(recommendation['suggestions'][:3], 1):
                print(f"║   {i}. {suggestion}")
        
        print("║")
        print("║ 🎭 선택 옵션:")
        for option in recommendation['user_options']:
            print(f"║   {option}")
        
        print("╚══════════════════════════════════════════════════════════════════╝")
        
        user_choice = input("\n선택하세요 [1-3]: ")
        
        return {
            "recommendation": recommendation["recommendation"],
            "user_choice": user_choice,
            "suggestions": recommendation.get("suggestions", []),
            "performance_score": recommendation.get("performance_score", 0)
        }

def generate_optimization_suggestions(results: dict, stage: int) -> list:
    """성능 기반 최적화 제안 생성"""
    suggestions = []
    
    # 메모리 관련
    if results.get("gpu_memory_gb", 0) > 14:
        suggestions.append("배치 크기 축소 (현재 메모리 사용량 높음)")
    
    # 검출 성능 관련
    detection_map = results.get("detection_mAP_0.5", 0)
    if detection_map < get_stage_target(stage, "detection.mAP_0.5") * 0.8:
        suggestions.append("검출 모델 하이퍼파라미터 튜닝 (학습률, 앵커 크기)")
        suggestions.append("데이터 증강 강화 고려")
    
    # 분류 성능 관련  
    classification_acc = results.get("classification_accuracy", 0)
    if classification_acc < get_stage_target(stage, "classification.accuracy") * 0.8:
        suggestions.append("분류 모델 정규화 강화 (드롭아웃, 레이블 스무딩)")
        suggestions.append("클래스 불균형 대응 강화 (가중 샘플링)")
    
    # 시스템 성능 관련
    if results.get("data_loading_time_s", 0) > 3:
        suggestions.append("데이터 로더 워커 수 조정")
        suggestions.append("LMDB 캐시 활용 확대")
    
    return suggestions
```

D-12) CLI/오버라이드

- python -m src.train --cfg config.yaml train.batch_size=128 dataloader.num_workers=12 train.resume=last
- 규칙: CLI > config.yaml > .env(.env는 API/서빙용)

[Annex 1 — Auto Batch Tuner (필수)]

- 입력: init_bs, min=1, max=init_bs*4 (메모리 여유 시 가변)
- 알고리즘: 워밍업(몇 step) → 시도 → OOM 시 empty_cache→fp16 강제→accum↑→batch↓ → 성공 시 상향 탐색
- 출력: 확정 batch_size; TB(perf/auto_batch_found), 로그에 후보/시도/결정 사유

[Annex 2 — Worker Autotune (필수)]

- 후보: [4,8,12,16] (config에서 가져와도 됨)
- 각 후보로 50~100 step data_time 평균 측정 → 최솟값 선택
- TB(perf/num_workers), 로그에 후보별 결과와 선택값

[Annex 3 — AMP/TF32/compile/channels_last 정책 (강제)]

- AMP: autocast+Scaler, "auto"면 bf16 우선 → 미지원 시 fp16
- TF32: allow_tf32=True + matmul_precision("high")
- channels_last: 모델·입력 모두 메모리 포맷 적용(가능할 때만)
- torch.compile: 3가지 모드만 지원 ("none", "reduce-overhead", "max-autotune"), 실패 시 경고 후 일반 모델 사용

[Annex 5 — EMA (옵션)]

- enabled=True면 timm.utils.ModelEmaV2 또는 동등 래퍼 사용(decay≈0.9998)
- 평가/저장은 EMA 가중치 기준(옵션: raw/ema 둘 다 기록)

[Annex 6 — OOM 폴백 상태 머신 (강제)]

- 순서/로그/TB 태그를 위 “D-8” 규칙대로 구현

[Annex 7 — 성능 로깅 스키마 (강제)]

- 스텝별: step, epoch, loss, lr, data_time, compute_time, throughput(img/s), cuda_mem(MB)
- TB 태그: train/loss, train/lr, perf/data_time, perf/compute_time, perf/throughput, perf/cuda_mem
- 검증: val/loss, val/accuracy, val/macro_f1, (옵션) val/top5

[Annex 8 — 체크포인트 스키마 (강제)]

- best.pt / last.pt
- 포함: model_state, optimizer_state, scheduler_state, scaler_state, epoch, best_metric, config_snapshot
- resume 로딩 시 key 불일치/백본 차이 검증 및 친절한 에러

[실행 명령 예시]
$ bash scripts/core/setup_venv.sh
$ bash scripts/train.sh

# 재개

$ python -m src.train --cfg config.yaml train.resume=last

# 하이퍼 오버라이드

$ python -m src.train --cfg config.yaml train.batch_size=128 dataloader.num_workers=12

# 빠른 스모크(1~2 epoch)로 파이프라인 점검 권장

[성공 판정(이 파트 완료 기준)]

- exp_dir/logs/train.out|err 생성 및 학습 로그가 위 포맷으로 출력됨
- exp_dir/checkpoints/{last.pt, best.pt} 생성
- exp_dir/reports/metrics.json 갱신(최소 acc/macro_f1/val_loss 포함)
- exp_dir/tb/ 하위에 TensorBoard 이벤트 파일 생성
- Stage 4 완료시 exp_dir/reports/final_test_results.json 생성
- OOM이나 배치/워커 조정 로그가 남고, 실패해도 친절한 메시지로 종료

### Stage 1 시간 캡 성공 판정(명시)

- config.yaml의 `data.progressive_validation.stage_1`에 다음이 모두 충족되면 “성공(파이프라인 검증 통과)”로 간주한다:
  - `allow_success_on_time_cap: true`
  - 실제 실행 시간이 `time_limit_hours`에 도달하여 조기 종료
  - 처리 샘플 수가 `min_samples_required` 이상
  - 커버된 클래스 수가 `min_class_coverage` 이상
  - 필수 체크(mandatory_checks): pipeline_complete, classification_model_loading(OK)
  - 지표 측정이 가능하면 기록하되, 미달이어도 시간 캡 성공은 유지

## 🎯 **PART_D 핵심 업데이트 완료**

### ✅ **조건부 Two-Stage 학습 파이프라인**
- **단일 약품**: EfficientNetV2-S 직접 분류 (384px, 5000클래스)
- **조합 약품**: YOLOv11m 검출 → 크롭 → EfficientNetV2-S 분류  
- **조건부 전환**: 사용자 선택 기반 파이프라인 분기
- **Commercial 아키텍처**: 함수 기반 새 컴포넌트 추가 완료

### ✅ **RTX 5080 16GB 최적화**
- **Auto Batch**: OOM 감지 → 자동 배치 크기 조정 (검출 16, 분류 64)
- **Mixed Precision**: AMP auto (bfloat16 > fp16), TF32 활성화
- **Memory Format**: channels_last 최적화, torch.compile("reduce-overhead")
- **OOM 폴백**: empty_cache → fp16 강제 → grad_accum↑ → batch↓

### ✅ **128GB RAM 활용**
- **Worker Autotune**: 16 스레드 최적 활용, data_time 최소화
- **Prefetch**: prefetch_factor=8, persistent_workers=True
- **Memory Pin**: pin_memory_device="cuda" RTX 5080 직접 핀

### ✅ **안정성 & 가시성**
- **체크포인트**: best.pt/last.pt 자동 저장, 재개 지원
- **TensorBoard**: 실시간 성능 모니터링 (mAP, accuracy, loss, throughput)
- **로깅**: 단계별 data_time, compute_time, VRAM 사용량

**✅ PART_D 완료: RTX 5080 최적화된 조건부 Two-Stage 학습 파이프라인**
