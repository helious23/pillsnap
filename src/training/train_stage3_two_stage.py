#!/usr/bin/env python3
"""
Stage 3 Two-Stage Pipeline 학습기

Detection + Classification 통합 학습:
- YOLOv11x Detection (Stage 4 준비용 기능 검증)
- EfficientNetV2-L Classification (높은 성능 유지)  
- 교차 학습 (Interleaved Training)
- RTX 5080 최적화 (Mixed Precision, torch.compile)
- 목표: Detection mAP@0.5 ≥ 0.30, Classification Accuracy ≥ 85%
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler
import torch.amp
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import pandas as pd
from dataclasses import dataclass

from src.models.classifier_efficientnetv2 import PillSnapClassifier, create_pillsnap_classifier
from src.models.detector_yolo11m import PillSnapYOLODetector, create_pillsnap_detector
from src.data.dataloader_manifest_training import ManifestDataset, ManifestTrainingDataLoader
from src.training.memory_monitor_gpu_usage import GPUMemoryMonitor  
from src.evaluation.evaluate_classification_metrics import ClassificationMetricsEvaluator
from src.evaluation.evaluate_detection_metrics import DetectionMetricsEvaluator
from src.utils.core import PillSnapLogger, load_config
from src.training.tensorboard_integration import patch_trainer_with_tensorboard


@dataclass
class TwoStageTrainingConfig:
    """Two-Stage 학습 설정 - RTX 5080 Native Linux 최적화"""
    
    # 학습 기본 설정
    max_epochs: int = 20
    learning_rate_classifier: float = 2e-4
    learning_rate_detector: float = 1e-3
    batch_size: int = 16
    
    # 교차 학습 설정 - 분류기 중심
    interleaved_training: bool = True
    classifier_epochs_per_cycle: int = 1  # 에포크당 1회 학습 (정상 동작)
    detector_epochs_per_cycle: int = 1    # 검출기도 1회 학습
    
    # Native Linux 최적화 설정
    mixed_precision: bool = True
    torch_compile: bool = True
    channels_last: bool = True
    
    # 타겟 지표
    target_classification_accuracy: float = 0.85
    target_detection_map: float = 0.30


class Stage3TwoStageTrainer:
    """Stage 3 Two-Stage Pipeline 학습기"""
    
    def __init__(
        self,
        config_path: str = "config.yaml",
        manifest_train: str = "artifacts/stage3/manifest_train.csv",
        manifest_val: str = "artifacts/stage3/manifest_val.csv",
        device: str = "cuda",
        resume_checkpoint: str = None
    ):
        self.device = torch.device(device)
        self.logger = PillSnapLogger(__name__)
        
        # 설정 로드
        self.config = load_config(config_path)
        self.manifest_train = Path(manifest_train)
        self.manifest_val = Path(manifest_val)
        
        # Stage 3 설정 확인
        self.stage_config = self.config['progressive_validation']['stage_configs']['stage_3']
        
        # 학습 설정
        self.training_config = TwoStageTrainingConfig()
        self.seed = 42
        torch.manual_seed(self.seed)
        
        # Resume 체크포인트 경로 저장
        self._resume_checkpoint_path = resume_checkpoint
        
        # torch.compile 워커 수 설정 (Smoke Test 검증된 8개)
        os.environ["TORCH_COMPILE_MAX_PARALLEL_COMPILE_JOBS"] = "8"
        
        # 모델 및 도구 초기화
        self.classifier = None
        self.detector = None
        self.classification_dataloader = None
        self.detection_dataloader = None
        self.memory_monitor = GPUMemoryMonitor()
        
        # Detection 실측치 추적
        self.last_detection_map = 0.0
        
        # DataLoader 캐싱 (매 epoch마다 재생성 방지)
        self.train_loader_cache = None
        self.val_loader_cache = None
        
        # 학습 상태
        self.best_classification_accuracy = 0.0
        self.best_classification_top5_accuracy = 0.0
        self.best_detection_map = 0.0
        self.training_history = []
        
        self.logger.info("Stage 3 Two-Stage Pipeline Trainer 초기화 완료")
        self.logger.info(f"목표 - Classification: {self.training_config.target_classification_accuracy:.1%}")
        self.logger.info(f"목표 - Detection mAP@0.5: {self.training_config.target_detection_map:.1%}")
    
    def setup_data_loaders(self) -> None:
        """데이터 로더 설정"""
        
        try:
            self.logger.info("데이터 로더 설정 시작...")
            
            # Manifest 파일 확인
            if not self.manifest_train.exists():
                raise FileNotFoundError(f"학습 manifest 파일이 없습니다: {self.manifest_train}")
            if not self.manifest_val.exists():
                raise FileNotFoundError(f"검증 manifest 파일이 없습니다: {self.manifest_val}")
            
            # Classification 데이터 로더 (Single + Combination crop)
            train_manifest_df = pd.read_csv(self.manifest_train)
            val_manifest_df = pd.read_csv(self.manifest_val)
            
            self.logger.info(f"학습 데이터: {len(train_manifest_df)} 샘플")
            self.logger.info(f"검증 데이터: {len(val_manifest_df)} 샘플")
            
            # Single/Combination 비율 확인 (Manifest는 image_type 컬럼 사용)
            train_single = train_manifest_df[train_manifest_df['image_type'] == 'single']
            train_combo = train_manifest_df[train_manifest_df['image_type'] == 'combination']
            
            self.logger.info(f"학습 - Single: {len(train_single)} ({len(train_single)/len(train_manifest_df):.1%})")
            self.logger.info(f"학습 - Combination: {len(train_combo)} ({len(train_combo)/len(train_manifest_df):.1%})")
            
            # Classification 데이터로더 (전체 데이터)
            self.classification_dataloader = ManifestTrainingDataLoader(
                manifest_train_path=str(self.manifest_train),
                manifest_val_path=str(self.manifest_val),
                batch_size=self.training_config.batch_size,
                image_size=384,  # EfficientNetV2-L
                num_workers=8,  # Native Linux 최적화
                task="classification"
            )
            
            # Detection 데이터로더 (Combination만)
            # Combination 데이터만 별도 manifest 생성
            combo_train_path = "artifacts/stage3/manifest_train_combo.csv"
            combo_val_path = "artifacts/stage3/manifest_val_combo.csv"
            
            train_combo.to_csv(combo_train_path, index=False)
            val_combo = val_manifest_df[val_manifest_df['image_type'] == 'combination']
            val_combo.to_csv(combo_val_path, index=False)
            
            self.detection_dataloader = ManifestTrainingDataLoader(
                manifest_train_path=combo_train_path,
                manifest_val_path=combo_val_path,
                batch_size=max(8, self.training_config.batch_size // 2),  # Detection은 더 적은 배치
                image_size=640,  # YOLOv11x
                num_workers=4,
                task="detection"
            )
            
            self.logger.info("데이터 로더 설정 완료")
            
        except Exception as e:
            self.logger.error(f"데이터 로더 설정 실패: {e}")
            raise
    
    def setup_models(self) -> None:
        """모델 설정"""
        
        try:
            self.logger.info("모델 설정 시작...")
            
            # 클래스 수 확인 (Manifest는 mapping_code 컬럼 사용)
            train_manifest_df = pd.read_csv(self.manifest_train)
            num_classes = train_manifest_df['mapping_code'].nunique()
            self.logger.info(f"분류 클래스 수: {num_classes}")
            
            # Classification 모델 (EfficientNetV2-L)
            self.classifier = create_pillsnap_classifier(
                num_classes=num_classes,
                model_name="efficientnetv2_l", 
                pretrained=True,
                device=self.device
            )
            
            # Detection 모델 (YOLOv11x) - 1개 클래스 (pill)
            self.detector = create_pillsnap_detector(
                num_classes=1,  # 약품 검출용
                model_size="yolo11x",  # Stage 3+ 대형 모델
                input_size=640,
                device=self.device
            )
            
            # 최적화 적용
            if self.training_config.channels_last:
                self.classifier = self.classifier.to(memory_format=torch.channels_last)
                
            if self.training_config.torch_compile:
                # reduce-overhead 모드: 컴파일 시간 단축, 안정적 성능
                self.classifier = torch.compile(self.classifier, mode='reduce-overhead')
                self.logger.info("torch.compile 최적화 적용 (reduce-overhead 모드)")
            
            self.logger.info("모델 설정 완료")
            
        except Exception as e:
            self.logger.error(f"모델 설정 실패: {e}")
            raise
    
    def setup_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """옵티마이저 및 스케줄러 설정"""
        
        # ConfigProvider에서 learning rate 가져오기 (런타임 오버라이드 지원)
        from src.utils.core import config_provider
        
        lr_classifier = config_provider.get('train.lr_classifier', self.training_config.learning_rate_classifier)
        lr_detector = config_provider.get('train.lr_detector', self.training_config.learning_rate_detector)
        
        # CLI 인자로 오버라이드 확인
        if hasattr(self, '_lr_classifier_override') and self._lr_classifier_override:
            lr_classifier = self._lr_classifier_override
            self.logger.info(f"Classifier LR override: {lr_classifier}")
        
        if hasattr(self, '_lr_detector_override') and self._lr_detector_override:
            lr_detector = self._lr_detector_override
            self.logger.info(f"Detector LR override: {lr_detector}")
        
        # 옵티마이저 생성
        classifier_optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=lr_classifier,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        detector_optimizer = optim.AdamW(
            self.detector.parameters(),
            lr=lr_detector,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 스케줄러 생성 (CosineAnnealingWarmRestarts)
        self.scheduler_cls = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            classifier_optimizer,
            T_0=10,  # 첫 번째 restart까지 epoch 수
            T_mult=2,  # restart 주기 배수
            eta_min=1e-6  # 최소 learning rate
        )
        
        self.scheduler_det = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            detector_optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.logger.info(f"Optimizers 설정 완료:")
        self.logger.info(f"  - Classifier LR: {lr_classifier}")
        self.logger.info(f"  - Detector LR: {lr_detector}")
        self.logger.info(f"  - Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
        
        return classifier_optimizer, detector_optimizer
    
    def train_classification_epoch(
        self, 
        optimizer: optim.Optimizer, 
        scaler: GradScaler,
        epoch: int
    ) -> Dict[str, float]:
        """분류기 한 에포크 학습"""
        
        self.classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # DataLoader 캐싱: 첫 번째 epoch에서만 생성
        if self.train_loader_cache is None:
            self.train_loader_cache = self.classification_dataloader.get_train_loader()
            self.logger.info("Train DataLoader 캐시 생성 완료")
        
        train_loader = self.train_loader_cache
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            if self.training_config.channels_last:
                images = images.to(memory_format=torch.channels_last)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.training_config.mixed_precision):
                outputs = self.classifier(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 20 == 0:  # 20 배치마다 출력 (더 자주)
                self.logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
                # TensorBoard 로깅
                if hasattr(self, 'log_tb_classification_batch'):
                    current_lr = optimizer.param_groups[0]['lr']
                    batch_accuracy = correct / total if total > 0 else 0
                    self.log_tb_classification_batch(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        total_batches=len(train_loader),
                        loss=loss.item(),
                        accuracy=batch_accuracy,
                        lr=current_lr
                    )
                
            # 처음 몇 개 배치는 더 자주 출력
            if batch_idx < 10:
                self.logger.info(f"초기 배치 {batch_idx} | Loss: {loss.item():.4f}")
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Classification 학습 완료 로깅
        self.logger.info(f"✅ Classification Epoch {epoch} 완료 | Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.4f}")
        
        return {
            'classification_loss': avg_loss,
            'classification_accuracy': accuracy
        }
    
    def train_detection_epoch(
        self,
        optimizer: optim.Optimizer,
        epoch: int
    ) -> Dict[str, float]:
        """검출기 한 에폿크 학습 - Ultralytics YOLO.train() 사용 (누적 학습 + 실제 메트릭)"""
        
        import subprocess
        import tempfile
        import time
        
        # 검증 주기 상수
        VAL_PERIOD = 3  # 3 에폭마다 검증
        YOLO_PROJECT = '/home/max16/pillsnap/artifacts/yolo'
        YOLO_NAME = 'stage3'
        
        try:
            # 모델 경로 및 resume 설정 (개선된 로직)
            last_pt_path = Path(YOLO_PROJECT) / YOLO_NAME / 'weights' / 'last.pt'
            best_pt_path = Path(YOLO_PROJECT) / YOLO_NAME / 'weights' / 'best.pt'
            
            # YOLO 모델 초기화 전략
            if epoch == 1:
                # 첫 에포크: resume 체크포인트 또는 pretrained 사용
                if hasattr(self, '_resume_yolo_checkpoint') and self._resume_yolo_checkpoint:
                    # --resume으로 지정된 체크포인트 사용
                    model_path = self._resume_yolo_checkpoint
                    resume = True
                    self.logger.info(f"🔄 Detection Resume: {model_path}")
                elif last_pt_path.exists():
                    # 이전 실행의 last.pt 존재시 사용
                    model_path = str(last_pt_path)
                    resume = True
                    self.logger.info(f"🔄 Detection 이전 학습 재개: {model_path}")
                else:
                    # 처음부터 시작
                    model_path = 'yolo11x.pt'
                    resume = False
                    self.logger.info(f"🆕 Detection 학습 시작: {model_path}")
            else:
                # 이후 에포크: 항상 last.pt에서 이어서 학습
                if last_pt_path.exists():
                    model_path = str(last_pt_path)
                    resume = True  # 항상 True (학습 지속)
                    self.logger.info(f"✅ Detection 학습 지속: {model_path}")
                else:
                    # Fallback: best.pt 또는 pretrained
                    if best_pt_path.exists():
                        model_path = str(best_pt_path)
                        resume = True
                        self.logger.warning(f"⚠️ last.pt 없음, best.pt 사용: {model_path}")
                    else:
                        model_path = 'yolo11x.pt'
                        resume = False
                        self.logger.warning(f"⚠️ 체크포인트 없음, pretrained 사용: {model_path}")
            
            self.logger.info(f"🎯 Detection 학습 시작 (Epoch {epoch})")
            
            # YOLO 데이터셋 설정 파일 생성
            dataset_yaml = self._create_yolo_dataset_config()
            
            # 검증 여부 결정
            do_validation = (epoch % VAL_PERIOD == 0)
            
            # 임시 로그 파일
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.log', delete=False) as temp_log:
                temp_log_path = temp_log.name
            
            # YOLO 학습을 subprocess로 실행하여 출력 캡처
            cmd = [
                sys.executable, '-c',
                f"""
import sys
sys.path.insert(0, '/home/max16/pillsnap')
from ultralytics import YOLO

model = YOLO('{model_path}')
results = model.train(
    data='{dataset_yaml}',
    epochs=1,
    batch={min(8, self.training_config.batch_size)},
    imgsz=640,
    device='{self.device.type}',
    save=True,  # 항상 저장 (last.pt, best.pt)
    save_period=1,  # 매 에포크마다 저장
    verbose=True,
    workers=4,
    rect=False,
    cache=False,
    plots=False,
    exist_ok=True,
    project='{YOLO_PROJECT}',
    name='{YOLO_NAME}',
    patience=0,  # Early stopping 비활성화
    val={do_validation},
    resume={resume},  # 동적으로 설정된 resume 값 사용
    deterministic=False,
    single_cls=False,
    optimizer='auto',
    seed=0,
    close_mosaic=10,
    copy_paste=0.0,
    auto_augment=None
)

# 결과 출력
if hasattr(results, 'results_dict'):
    print("RESULTS_DICT:", results.results_dict)
if hasattr(results, 'metrics'):
    print("METRICS:", results.metrics)
"""
            ]
            
            self.logger.info(f"YOLO 학습 subprocess 실행 중... (val={do_validation}, resume={resume})")
            
            # subprocess 실행하고 출력 실시간 로깅
            with open(temp_log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # 실시간으로 출력 읽기
                for line in iter(process.stdout.readline, ''):
                    if line:
                        line = line.strip()
                        log_file.write(line + '\n')
                        log_file.flush()
                        
                        # 중요한 로그만 필터링해서 출력
                        if any(keyword in line for keyword in ['box_loss', 'cls_loss', 'dfl_loss', 'Epoch', 'GPU', 'images']):
                            self.logger.info(f"[YOLO] {line}")
                            # 모니터링 로그 파일에도 쓰기
                            with open('/tmp/mini_test.log', 'a') as monitor_log:
                                monitor_log.write(f"{time.strftime('%H:%M:%S')} | INFO     | [YOLO] {line}\n")
                                monitor_log.flush()
                
                process.wait()
            
            # 임시 파일 삭제
            os.unlink(temp_log_path)
            
            # results.csv에서 실제 메트릭 읽기
            results_csv_path = Path(YOLO_PROJECT) / YOLO_NAME / 'results.csv'
            
            # 메트릭 컬럼 매핑 (버전 호환성)
            METRIC_COLUMNS = {
                'mAP': ['metrics/mAP50(B)', 'metrics/mAP50', 'mAP50'],
                'precision': ['metrics/precision(B)', 'metrics/precision', 'precision'],
                'recall': ['metrics/recall(B)', 'metrics/recall', 'recall'],
                'box_loss': ['train/box_loss', 'box_loss'],
                'cls_loss': ['train/cls_loss', 'cls_loss'],
                'dfl_loss': ['train/dfl_loss', 'dfl_loss']
            }
            
            # 기본값 설정
            val_map = 0.0
            precision = 0.0
            recall = 0.0
            box_loss = None
            cls_loss = None
            dfl_loss = None
            
            # CSV 읽기 시도 (재시도 포함)
            for attempt in range(3):
                try:
                    if results_csv_path.exists():
                        df = pd.read_csv(results_csv_path)
                        if not df.empty:
                            last_row = df.iloc[-1]
                            
                            # 각 메트릭을 컬럼 후보 리스트에서 찾기
                            for metric_name, column_candidates in METRIC_COLUMNS.items():
                                for col in column_candidates:
                                    if col in last_row:
                                        value = last_row[col]
                                        if pd.notna(value):
                                            if metric_name == 'mAP':
                                                val_map = float(value)
                                            elif metric_name == 'precision':
                                                precision = float(value)
                                            elif metric_name == 'recall':
                                                recall = float(value)
                                            elif metric_name == 'box_loss':
                                                box_loss = float(value)
                                            elif metric_name == 'cls_loss':
                                                cls_loss = float(value)
                                            elif metric_name == 'dfl_loss':
                                                dfl_loss = float(value)
                                            break
                            
                            self.logger.info(f"✅ CSV 메트릭 로드 성공: {results_csv_path}")
                            break
                    else:
                        self.logger.warning(f"results.csv 없음: {results_csv_path}")
                        
                except Exception as e:
                    if attempt < 2:
                        time.sleep(0.5)  # 짧은 대기 후 재시도
                    else:
                        self.logger.warning(f"CSV 읽기 실패: {e}")
            
            # total_loss 계산
            if box_loss is not None and cls_loss is not None and dfl_loss is not None:
                total_loss = (box_loss + cls_loss + dfl_loss) / 3.0
            else:
                total_loss = None
                self.logger.warning("Loss 값을 CSV에서 읽지 못함, None으로 설정")
            
            # 상세한 Detection 메트릭 로깅
            self.logger.info(f"✅ Detection Epoch {epoch} 완료")
            if box_loss is not None:
                self.logger.info(f"[Detection Epoch {epoch}] box_loss: {box_loss:.4f} | cls_loss: {cls_loss:.4f} | dfl_loss: {dfl_loss:.4f}")
            if total_loss is not None:
                self.logger.info(f"[Detection Epoch {epoch}] Total Loss: {total_loss:.4f} | mAP@0.5: {val_map:.3f}")
            
            # 실측치 저장 (validate_models에서 사용)
            self.last_detection_map = val_map
            
            # 모니터링 파서용 표준 태그 (DET_SUMMARY)
            # 값 먼저 정하고 포맷팅
            box_loss_val = 0.0 if box_loss is None else float(box_loss)
            cls_loss_val = 0.0 if cls_loss is None else float(cls_loss)
            dfl_loss_val = 0.0 if dfl_loss is None else float(dfl_loss)
            total_loss_val = 0.0 if total_loss is None else float(total_loss)
            
            self.logger.info(
                f"DET_SUMMARY | epoch={epoch} | "
                f"box_loss={box_loss_val:.4f} | "
                f"cls_loss={cls_loss_val:.4f} | "
                f"dfl_loss={dfl_loss_val:.4f} | "
                f"total_loss={total_loss_val:.4f} | "
                f"mAP={val_map:.4f}"
            )
            
            # DET_DETAIL 로그 (confidence는 0.0으로 설정, 나중에 tuner가 덮어쓸 수 있음)
            self.logger.info(
                f"DET_DETAIL | recall={recall:.3f} | precision={precision:.3f} | "
                f"det_conf=0.000 | cls_conf=0.000 | single_conf=0.000 | combo_conf=0.000"
            )
            
            return {
                'detection_loss': total_loss if total_loss is not None else 0.0,
                'detection_map': val_map,
                'detection_precision': precision,
                'detection_recall': recall
            }
            
        except Exception as e:
            self.logger.warning(f"Detection 학습 에러 - 손상된 파일이나 일시적 문제로 스킵: {e}")
            # 에러 유형별 처리
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ['truncated', 'corrupt', 'bad image', 'decode']):
                self.logger.info("이미지 파일 관련 에러 - 다음 epoch에서 자동으로 스킵됩니다")
            elif 'cuda' in error_msg or 'memory' in error_msg:
                self.logger.warning("GPU 메모리 부족 - 배치 크기를 줄여보세요")
            else:
                self.logger.warning(f"기타 Detection 에러: {error_msg}")
            
            # 기본 메트릭 반환 (학습 계속 진행)
            return {
                'detection_loss': 0.0,
                'detection_map': 0.0,
                'detection_precision': 0.0,
                'detection_recall': 0.0
            }
    
    def _create_yolo_dataset_config(self) -> Path:
        """YOLO 데이터셋 설정 파일 생성 - 실제 데이터 구조에 맞게 조정"""
        import yaml
        
        # YOLO 설정 파일 경로
        config_dir = Path("/home/max16/pillsnap_data/yolo_configs")
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "stage3_detection.yaml"
        
        # YOLO 호환 데이터셋 디렉토리 생성
        yolo_dataset_root = config_dir / "yolo_dataset" 
        yolo_images_dir = yolo_dataset_root / "images"
        yolo_labels_dir = yolo_dataset_root / "labels"
        
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 심볼릭 링크들 정리
        for f in yolo_images_dir.glob("*"):
            if f.is_file() or f.is_symlink():
                try:
                    f.unlink()
                except Exception:
                    pass
        for f in yolo_labels_dir.glob("*"):
            if f.is_file() or f.is_symlink():
                try:
                    f.unlink()
                except Exception:
                    pass
        
        # 이미지와 라벨 심볼릭 링크 생성 (매칭되는 것만 + 유효한 파일만)
        base_path = Path("/home/max16/pillsnap_data/train/images/combination")
        label_path = Path("/home/max16/pillsnap_data/train/labels/combination_yolo")
        
        linked_count = 0
        skipped_count = 0
        
        for ts_dir in base_path.glob("TS_*_combo"):
            if not ts_dir.is_dir():
                continue
                
            for k_dir in ts_dir.iterdir():
                if not k_dir.is_dir():
                    continue
                    
                # 이미지 파일들을 찾고 매칭되는 라벨이 있는 것만 링크 (유효성 검사 포함)
                for img_file in k_dir.glob("*_0_2_0_2_*.png"):
                    label_file = label_path / f"{img_file.stem}.txt"
                    
                    if label_file.exists():
                        # 이미지 파일 유효성 검사
                        try:
                            # 파일 크기 체크 (손상된 파일은 보통 매우 작음)
                            if img_file.stat().st_size < 100:  
                                self.logger.debug(f"스킵: 너무 작은 파일 {img_file.name}")
                                skipped_count += 1
                                continue
                            
                            # 심볼릭 링크 생성
                            img_link = yolo_images_dir / img_file.name
                            label_link = yolo_labels_dir / label_file.name
                            
                            if not img_link.exists():
                                img_link.symlink_to(img_file.absolute())
                            if not label_link.exists():
                                label_link.symlink_to(label_file.absolute())
                                
                            linked_count += 1
                            
                        except (OSError, IOError) as e:
                            # 파일 접근 오류 시 스킵
                            self.logger.debug(f"스킵: 파일 접근 오류 {img_file.name}: {e}")
                            skipped_count += 1
                            continue
        
        self.logger.info(f"YOLO 데이터셋 준비: {linked_count}개 이미지-라벨 쌍 링크 생성")
        if skipped_count > 0:
            self.logger.info(f"손상되거나 문제있는 파일 {skipped_count}개 스킵됨")
        
        # YOLO 데이터셋 설정
        config = {
            'path': str(yolo_dataset_root),  # YOLO 데이터셋 루트
            'train': 'images',  # 이미지 디렉토리 (상대 경로)
            'val': 'images',    # 검증 이미지 디렉토리 (같은 경로 사용)
            'names': {0: 'pill'},  # 클래스 이름
            'nc': 1  # 클래스 개수
        }
        
        # YAML 파일 생성
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.logger.info(f"YOLO 데이터셋 설정 생성: {config_path}")
        self.logger.info(f"  - 데이터셋 경로: {yolo_dataset_root}")
        self.logger.info(f"  - 이미지: {len(list(yolo_images_dir.glob('*.png')))}개")
        self.logger.info(f"  - 라벨: {len(list(yolo_labels_dir.glob('*.txt')))}개")
        
        return config_path
    
    def validate_models(self) -> Dict[str, float]:
        """모델 검증 - Classification만 실제 평가, Detection은 train_detection_epoch의 실측치 사용"""
        
        results = {}
        
        # Classification 검증 (실제 평가)
        self.classifier.eval()
        correct = 0
        correct_top5 = 0
        total = 0
        
        # DataLoader 캐싱: 첫 번째 validation에서만 생성
        if self.val_loader_cache is None:
            self.val_loader_cache = self.classification_dataloader.get_val_loader()
            self.logger.info("Validation DataLoader 캐시 생성 완료")
        
        val_loader = self.val_loader_cache
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.training_config.channels_last:
                    images = images.to(memory_format=torch.channels_last)
                
                with torch.amp.autocast('cuda', enabled=self.training_config.mixed_precision):
                    outputs = self.classifier(images)
                
                # Top-1 accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Top-5 accuracy
                _, pred_top5 = outputs.topk(5, 1, True, True)
                pred_top5 = pred_top5.t()
                correct_expanded = labels.view(1, -1).expand_as(pred_top5)
                correct_top5 += pred_top5.eq(correct_expanded).sum().item()
        
        results['val_classification_accuracy'] = correct / total
        results['val_classification_top5_accuracy'] = correct_top5 / total
        
        # 검증 결과 즉시 로깅
        self.logger.info(f"📊 Validation Results:")
        self.logger.info(f"  - Classification Accuracy: {results['val_classification_accuracy']:.4f} ({correct}/{total})")
        self.logger.info(f"  - Top-5 Accuracy: {results['val_classification_top5_accuracy']:.4f}")
        
        # Detection 메트릭은 train_detection_epoch()의 실측치 사용
        # 가장 최근 detection 메트릭 가져오기
        if hasattr(self, 'last_detection_map'):
            results['val_detection_map'] = self.last_detection_map
        else:
            # 초기값 또는 이전 체크포인트 값 사용
            results['val_detection_map'] = getattr(self, 'best_detection_map', 0.0)
        
        self.logger.info(f"  - Detection mAP@0.5: {results['val_detection_map']:.4f} (from training)")
        
        return results
    
    def train(self, start_epoch: int = 0) -> Dict[str, Any]:
        """Two-Stage 교차 학습 실행"""
        
        self.logger.info("Two-Stage Pipeline 학습 시작")
        self.current_epoch = start_epoch
        
        # 데이터 및 모델 설정
        self.setup_data_loaders()
        self.setup_models()
        
        # 옵티마이저를 self에 저장하여 체크포인트 저장/로드 가능하게 함
        self.optimizer_cls, self.optimizer_det = self.setup_optimizers()
        
        # Patience counter 초기화
        self.cls_patience_counter = 0
        self.det_patience_counter = 0
        
        # 체크포인트 로드 (모델 초기화 후에 실행)
        if hasattr(self, '_resume_checkpoint_path') and self._resume_checkpoint_path:
            start_epoch, _ = self.load_checkpoint(self._resume_checkpoint_path)
            self.logger.info(f"체크포인트에서 resume: epoch {start_epoch}")
        
        scaler = GradScaler(enabled=self.training_config.mixed_precision)
        
        start_time = time.time()
        
        for epoch in range(start_epoch + 1, self.training_config.max_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            epoch_results = {'epoch': epoch}
            
            # 교차 학습: Classification → Detection
            if self.training_config.interleaved_training:
                
                # Classification 학습
                for _ in range(self.training_config.classifier_epochs_per_cycle):
                    cls_results = self.train_classification_epoch(
                        self.optimizer_cls, scaler, epoch
                    )
                    epoch_results.update(cls_results)
                
                # Detection 학습  
                for _ in range(self.training_config.detector_epochs_per_cycle):
                    det_results = self.train_detection_epoch(
                        self.optimizer_det, epoch
                    )
                    epoch_results.update(det_results)
            
            # 검증
            val_results = self.validate_models()
            epoch_results.update(val_results)
            
            # 스케줄러 step
            if hasattr(self, 'scheduler_cls'):
                self.scheduler_cls.step()
                current_lr_cls = self.scheduler_cls.get_last_lr()[0]
                self.logger.info(f"📈 Classifier LR updated: {current_lr_cls:.2e}")
                
            if hasattr(self, 'scheduler_det'):
                self.scheduler_det.step()
                current_lr_det = self.scheduler_det.get_last_lr()[0]
                self.logger.info(f"📈 Detector LR updated: {current_lr_det:.2e}")
            
            # 최고 성능 업데이트 (epsilon 기준 적용)
            BEST_EPS = 0.001  # 0.1% 이상 개선 시 저장
            
            # Classification best 업데이트
            if val_results['val_classification_accuracy'] > self.best_classification_accuracy + BEST_EPS:
                self.best_classification_accuracy = val_results['val_classification_accuracy']
                self.best_classification_top5_accuracy = val_results['val_classification_top5_accuracy']
                self.save_checkpoint('classification', 'best')
                self.logger.info(f"✅ NEW BEST Classification: {self.best_classification_accuracy:.4f}")
                self.cls_patience_counter = 0  # Patience 초기화
            else:
                self.cls_patience_counter += 1
            
            # Detection best 업데이트
            if val_results['val_detection_map'] > self.best_detection_map + BEST_EPS:
                self.best_detection_map = val_results['val_detection_map']
                self.save_checkpoint('detection', 'best')
                self.logger.info(f"✅ NEW BEST Detection mAP: {self.best_detection_map:.4f}")
                self.det_patience_counter = 0  # Patience 초기화
            else:
                self.det_patience_counter += 1
            
            # 매 epoch마다 last 체크포인트 저장
            self.save_checkpoint('classification', 'last')
            self.save_checkpoint('detection', 'last')
            self.logger.info(f"💾 Saved last checkpoints - Cls: {val_results['val_classification_accuracy']:.4f}, Det: {val_results['val_detection_map']:.4f}")
            
            # Patience 기반 주기적 저장 (5 epochs 개선 없으면)
            PATIENCE_THRESHOLD = 5
            if self.cls_patience_counter >= PATIENCE_THRESHOLD:
                self.save_checkpoint('classification', 'periodic')
                self.logger.info(f"📦 Periodic save (patience={self.cls_patience_counter}) for Classification")
                self.cls_patience_counter = 0  # Reset after periodic save
                
            if self.det_patience_counter >= PATIENCE_THRESHOLD:
                self.save_checkpoint('detection', 'periodic')
                self.logger.info(f"📦 Periodic save (patience={self.det_patience_counter}) for Detection")
                self.det_patience_counter = 0  # Reset after periodic save
            
            # 수동 저장 트리거 확인
            from pathlib import Path
            save_flag_path = Path("artifacts/flags/save_now")
            if save_flag_path.exists():
                self.save_checkpoint('classification', 'manual')
                self.save_checkpoint('detection', 'manual')
                save_flag_path.unlink()
                self.logger.info("💾 Manual checkpoint saved due to save_now flag")
            
            # TensorBoard 에포크 로깅
            if hasattr(self, 'log_tb_classification_epoch'):
                self.log_tb_classification_epoch(
                    epoch=epoch,
                    train_loss=epoch_results.get('classification_loss', 0),
                    train_acc=epoch_results.get('classification_accuracy', 0),
                    val_loss=None,  # 현재 val loss 추적 안함
                    val_acc=val_results['val_classification_accuracy'],
                    val_top5=val_results['val_classification_top5_accuracy']
                )
            
            if hasattr(self, 'log_tb_detection_epoch'):
                self.log_tb_detection_epoch(
                    epoch=epoch,
                    box_loss=epoch_results.get('detection_box_loss', 0),
                    cls_loss=epoch_results.get('detection_cls_loss', 0),
                    dfl_loss=epoch_results.get('detection_dfl_loss', 0),
                    map50=val_results['val_detection_map'],
                    map50_95=None  # 현재 추적 안함
                )
            
            if hasattr(self, 'log_tb_system_metrics'):
                self.log_tb_system_metrics(epoch)
            
            # 로그 출력
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch:2d} | "
                f"Cls Acc: {val_results['val_classification_accuracy']:.3f} | "
                f"Top5 Acc: {val_results['val_classification_top5_accuracy']:.3f} | "
                f"Det mAP: {val_results['val_detection_map']:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )
            # 모니터링 파서용 표준 태그 로그
            self.logger.info(
                f"CLS_SUMMARY | epoch={epoch} | top1={val_results['val_classification_accuracy']:.4f} | "
                f"top5={val_results['val_classification_top5_accuracy']:.4f}"
            )
            self.logger.info(
                f"DET_SUMMARY | epoch={epoch} | map50={val_results['val_detection_map']:.4f}"
            )
            
            # 목표 달성 체크
            if (val_results['val_classification_accuracy'] >= self.training_config.target_classification_accuracy and 
                val_results['val_detection_map'] >= self.training_config.target_detection_map):
                self.logger.info("목표 성능 달성! 학습 조기 종료")
                break
            
            self.training_history.append(epoch_results)
        
        total_time = time.time() - start_time
        
        # 최종 결과
        final_results = {
            'training_completed': True,
            'total_training_time_minutes': total_time / 60,
            'best_classification_accuracy': self.best_classification_accuracy,
            'best_classification_top5_accuracy': self.best_classification_top5_accuracy,
            'best_detection_map': self.best_detection_map,
            'epochs_completed': getattr(self, 'current_epoch', start_epoch),
            'target_achieved': {
                'classification': self.best_classification_accuracy >= self.training_config.target_classification_accuracy,
                'detection': self.best_detection_map >= self.training_config.target_detection_map
            }
        }
        
        self.logger.info("Two-Stage Pipeline 학습 완료")
        self.logger.info(f"최고 Classification 정확도: {self.best_classification_accuracy:.3f}")
        self.logger.info(f"최고 Classification Top-5 정확도: {self.best_classification_top5_accuracy:.3f}")
        self.logger.info(f"최고 Detection mAP: {self.best_detection_map:.3f}")
        self.logger.info(f"총 학습 시간: {total_time/60:.1f}분")
        
        return final_results
    
    def save_checkpoint(self, model_type: str, checkpoint_type: str, force_save: bool = False) -> None:
        """
        체크포인트 저장 (개선된 정책)
        
        Args:
            model_type: 'classification' or 'detection'
            checkpoint_type: 'best', 'last', 'manual', 'periodic'
            force_save: 강제 저장 여부
        """
        
        try:
            checkpoint_dir = Path("artifacts/stage3/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # 주기적 저장 디렉토리 (epoch별 백업)
            if checkpoint_type == 'periodic':
                periodic_dir = checkpoint_dir / "periodic"
                periodic_dir.mkdir(exist_ok=True)
            
            if model_type == 'classification':
                if checkpoint_type == 'periodic':
                    checkpoint_path = periodic_dir / f"stage3_classification_epoch_{self.current_epoch:03d}.pt"
                else:
                    checkpoint_path = checkpoint_dir / f"stage3_classification_{checkpoint_type}.pt"
                
                # 저장할 데이터 준비
                checkpoint_data = {
                    'model_state_dict': self.classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer_cls.state_dict() if hasattr(self, 'optimizer_cls') else None,
                    'scheduler_state_dict': self.scheduler_cls.state_dict() if hasattr(self, 'scheduler_cls') else None,
                    'accuracy': self.best_classification_accuracy,
                    'top5_accuracy': self.best_classification_top5_accuracy,
                    'epoch': getattr(self, 'current_epoch', 0),
                    'training_history': getattr(self, 'training_history', {}),
                    'config': self.training_config,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                
            elif model_type == 'detection':
                if checkpoint_type == 'periodic':
                    checkpoint_path = periodic_dir / f"stage3_detection_epoch_{self.current_epoch:03d}.pt"
                else:
                    checkpoint_path = checkpoint_dir / f"stage3_detection_{checkpoint_type}.pt"
                
                # YOLO 모델 저장 (Ultralytics 방식)
                if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'save'):
                    self.detector.model.save(str(checkpoint_path))
                elif hasattr(self.detector, 'export'):
                    self.detector.export(format='torchscript', file=str(checkpoint_path))
                else:
                    # 대체 방법: 모델 state_dict 저장
                    checkpoint_data = {
                        'model_state_dict': self.detector.state_dict() if hasattr(self.detector, 'state_dict') else None,
                        'optimizer_state_dict': self.optimizer_det.state_dict() if hasattr(self, 'optimizer_det') else None,
                        'scheduler_state_dict': self.scheduler_det.state_dict() if hasattr(self, 'scheduler_det') else None,
                        'detection_map': self.best_detection_map,
                        'epoch': getattr(self, 'current_epoch', 0),
                        'training_history': getattr(self, 'detection_history', {}),
                        'config': self.training_config,
                        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                    }
                    torch.save(checkpoint_data, checkpoint_path)
            
            # 파일 크기 확인
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"💾 CHECKPOINT SAVED: {model_type} {checkpoint_type} → {checkpoint_path} ({file_size_mb:.1f}MB)")
            
            # 주기적 체크포인트 관리 (최대 5개 유지)
            if checkpoint_type == 'periodic':
                self._cleanup_old_periodic_checkpoints(periodic_dir, model_type, max_keep=5)
            
        except Exception as e:
            self.logger.warning(f"체크포인트 저장 실패: {e}")
    
    def _cleanup_old_periodic_checkpoints(self, periodic_dir: Path, model_type: str, max_keep: int = 5):
        """오래된 주기적 체크포인트 정리"""
        pattern = f"stage3_{model_type}_epoch_*.pt"
        checkpoints = sorted(periodic_dir.glob(pattern))
        
        if len(checkpoints) > max_keep:
            for old_checkpoint in checkpoints[:-max_keep]:
                old_checkpoint.unlink()
                self.logger.info(f"🗑️ Old checkpoint removed: {old_checkpoint.name}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, float]:
        """체크포인트 로드"""
        try:
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Classification model checkpoint
            if 'classification' in checkpoint_path:
                self.classifier.load_state_dict(checkpoint['model_state_dict'])
                if hasattr(self, 'optimizer_cls') and 'optimizer_state_dict' in checkpoint:
                    self.optimizer_cls.load_state_dict(checkpoint['optimizer_state_dict'])
                
                accuracy = checkpoint.get('accuracy', 0.0)
                epoch = checkpoint.get('epoch', 0)
                
                self.logger.info(f"Classification checkpoint loaded: {checkpoint_path}")
                self.logger.info(f"Resumed from epoch {epoch}, best accuracy: {accuracy:.3f}")
                return epoch, accuracy
                
            # Detection model checkpoint 
            elif 'detection' in checkpoint_path:
                if 'model_state_dict' in checkpoint and checkpoint['model_state_dict'] is not None:
                    self.detector.load_state_dict(checkpoint['model_state_dict'])
                    
                detection_map = checkpoint.get('detection_map', 0.0)
                epoch = checkpoint.get('epoch', 0)
                
                self.logger.info(f"Detection checkpoint loaded: {checkpoint_path}")
                self.logger.info(f"Resumed from epoch {epoch}, best mAP: {detection_map:.3f}")
                return epoch, detection_map
            
            return 0, 0.0
            
        except Exception as e:
            self.logger.error(f"체크포인트 로드 실패: {e}")
            return 0, 0.0


# TensorBoard 통합 패치 적용 (클래스 정의 후, 1회만)
patch_trainer_with_tensorboard(Stage3TwoStageTrainer)


def main():
    """메인 학습 함수 - 멀티프로세싱 워커에서 실행되지 않도록 보호"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 3 Two-Stage Pipeline Training")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--manifest-train", default="artifacts/stage3/manifest_train.csv", help="Train manifest path")
    parser.add_argument("--manifest-val", default="artifacts/stage3/manifest_val.csv", help="Val manifest path")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (default: 5 for smoke test)")
    parser.add_argument("--batch-size", type=int, default=20, help="Classification batch size (default: 20)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint path")
    parser.add_argument("--lr-classifier", type=float, help="Override classifier learning rate")
    parser.add_argument("--lr-detector", type=float, help="Override detector learning rate")
    parser.add_argument("--reset-best", action="store_true", help="Reset best metrics at start")
    
    args = parser.parse_args()
    
    trainer = Stage3TwoStageTrainer(
        config_path=args.config,
        manifest_train=args.manifest_train,
        manifest_val=args.manifest_val,
        device=args.device,
        resume_checkpoint=args.resume
    )
    
    # 명령행 인수로 설정 오버라이드
    trainer.training_config.max_epochs = args.epochs
    trainer.training_config.batch_size = args.batch_size
    
    # 하이퍼파라미터 오버라이드 (optimizer 생성 전에 설정)
    if args.lr_classifier:
        trainer._lr_classifier_override = args.lr_classifier
        trainer.training_config.learning_rate_classifier = args.lr_classifier
        trainer.logger.info(f"Classifier learning rate overridden: {args.lr_classifier}")
    
    if args.lr_detector:
        trainer._lr_detector_override = args.lr_detector
        trainer.training_config.learning_rate_detector = args.lr_detector
        trainer.logger.info(f"Detector learning rate overridden: {args.lr_detector}")
    
    # --reset-best 옵션 처리
    if args.reset_best:
        trainer.best_classification_accuracy = 0.0
        trainer.best_classification_top5_accuracy = 0.0
        trainer.best_detection_map = 0.0
        trainer.logger.info("✅ Reset best metrics due to --reset-best")
    
    print(f"🚀 Stage 3 Two-Stage 학습 시작")
    print(f"  에포크: {args.epochs}")
    print(f"  배치 크기: {args.batch_size}")
    
    results = trainer.train(start_epoch=0)
    print(f"✅ 학습 완료 - Classification: {results['best_classification_accuracy']:.3f}, Detection: {results['best_detection_map']:.3f}")


if __name__ == "__main__":
    # PyTorch 멀티프로세싱 환경에서는 기본 fork 방식 사용
    # spawn 방식은 DataLoader와 충돌 가능성 있음
    main()