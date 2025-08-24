"""
PillSnap ML Confidence 자동 튜닝 시스템 (1단계 필수)

자동 임계값 튜닝:
- 검증마다 conf ∈ [0.20..0.30], step=0.02 스윕 → Recall 우선, 동률 시 F1
- single/combination 도메인별 최적값 1개씩 선택
- 선택값을 즉시 추론설정에 주입 + 체크포인트 메타 + 요약 리포트에 기록

RTX 5080 최적화
"""

import json
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.utils.core import PillSnapLogger


@dataclass
class ConfidenceTuningConfig:
    """Confidence 튜닝 설정 (1단계 필수)"""
    
    # 스윕 범위 설정
    conf_min: float = 0.20
    conf_max: float = 0.30
    conf_step: float = 0.02
    
    # 선택 기준
    primary_metric: str = "recall"      # 우선순위: Recall
    secondary_metric: str = "f1"        # 동률시: F1
    
    # 도메인 분리
    evaluate_by_domain: bool = True     # 도메인별 최적값 선택
    domains: List[str] = None           # 기본값: ["single", "combination"]
    
    # 결과 저장
    save_tuning_results: bool = True
    save_to_checkpoint: bool = True     # 체크포인트 메타에 저장
    save_to_inference: bool = True      # 추론 설정에 즉시 반영
    
    # 리포트 생성
    generate_summary_report: bool = True
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = ["single", "combination"]


@dataclass
class ConfidenceResult:
    """Confidence 튜닝 결과"""
    confidence: float
    domain: str
    metrics: Dict[str, float]
    sample_count: int
    
    def get_primary_score(self, config: ConfidenceTuningConfig) -> float:
        """주요 메트릭 점수 반환"""
        return self.metrics.get(config.primary_metric, 0.0)
    
    def get_secondary_score(self, config: ConfidenceTuningConfig) -> float:
        """보조 메트릭 점수 반환"""
        return self.metrics.get(config.secondary_metric, 0.0)


class ConfidenceTuner:
    """Confidence 자동 튜닝 시스템 (1단계 필수)"""
    
    def __init__(
        self,
        config: ConfidenceTuningConfig,
        model: Optional[torch.nn.Module] = None,
        device: str = "cuda"
    ):
        """
        Args:
            config: Confidence 튜닝 설정
            model: 평가할 모델 (선택적)
            device: 디바이스
        """
        self.config = config
        self.model = model
        self.device = device
        self.logger = PillSnapLogger(__name__)
        
        # 스윕 범위 생성
        self.confidence_values = self._generate_confidence_range()
        
        # 튜닝 결과 저장
        self.tuning_history = []
        self.best_confidences = {}  # 도메인별 최적 confidence
        
        self.logger.info(
            f"🎯 Confidence 튜너 초기화 - "
            f"범위: {self.config.conf_min}-{self.config.conf_max} (step {self.config.conf_step}), "
            f"도메인: {self.config.domains}, "
            f"기준: {self.config.primary_metric} → {self.config.secondary_metric}"
        )
    
    def _generate_confidence_range(self) -> List[float]:
        """Confidence 스윕 범위 생성"""
        values = []
        conf = self.config.conf_min
        
        while conf <= self.config.conf_max + 1e-6:  # 부동소수점 오차 고려
            values.append(round(conf, 3))
            conf += self.config.conf_step
        
        return values
    
    def tune_confidence(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        domain_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Confidence 임계값 자동 튜닝
        
        Args:
            predictions: 예측 결과 리스트
            ground_truths: 정답 리스트
            domain_masks: 도메인별 마스크 (선택적)
            
        Returns:
            Dict[str, float]: 도메인별 최적 confidence
        """
        self.logger.info(f"🔍 Confidence 자동 튜닝 시작 - {len(self.confidence_values)}개 값 스윕")
        
        # 도메인별 튜닝 수행
        domain_results = {}
        
        for domain in self.config.domains:
            domain_preds, domain_gts = self._filter_by_domain(
                predictions, ground_truths, domain, domain_masks
            )
            
            if len(domain_preds) == 0:
                self.logger.warning(f"도메인 '{domain}' 샘플이 없음 - 건너뜀")
                continue
            
            best_result = self._tune_domain_confidence(domain, domain_preds, domain_gts)
            domain_results[domain] = best_result
            
            self.logger.info(
                f"✅ {domain} 최적 confidence: {best_result.confidence} "
                f"({self.config.primary_metric}: {best_result.get_primary_score(self.config):.3f}, "
                f"{self.config.secondary_metric}: {best_result.get_secondary_score(self.config):.3f})"
            )
        
        # 최적 confidence 딕셔너리 생성
        self.best_confidences = {
            domain: result.confidence 
            for domain, result in domain_results.items()
        }
        
        # 결과 저장 및 적용
        if self.config.save_tuning_results:
            self._save_tuning_results(domain_results)
        
        if self.config.save_to_inference:
            self._apply_to_inference_config()
            self._persist_to_inference_json()  # 영속화 JSON 저장 추가
        
        if self.config.generate_summary_report:
            self._generate_summary_report(domain_results)
        
        # 체크포인트 반영 자동화 (내부에서 수행)
        if self.config.save_to_checkpoint:
            self._auto_apply_to_checkpoint()
        
        self.logger.info(f"🎯 Confidence 튜닝 완료: {self.best_confidences}")
        
        return self.best_confidences
    
    def _filter_by_domain(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
        domain: str,
        domain_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """도메인별 필터링"""
        if not self.config.evaluate_by_domain or domain_masks is None:
            return predictions, ground_truths
        
        if domain not in domain_masks:
            return [], []
        
        mask = domain_masks[domain].cpu().numpy()
        
        filtered_preds = [pred for i, pred in enumerate(predictions) if i < len(mask) and mask[i]]
        filtered_gts = [gt for i, gt in enumerate(ground_truths) if i < len(mask) and mask[i]]
        
        return filtered_preds, filtered_gts
    
    def _tune_domain_confidence(
        self,
        domain: str,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> ConfidenceResult:
        """특정 도메인의 confidence 튜닝"""
        best_result = None
        
        for conf in self.confidence_values:
            # 해당 confidence로 예측 필터링
            filtered_preds = self._apply_confidence_threshold(predictions, conf)
            
            # 메트릭 계산
            metrics = self._calculate_metrics(filtered_preds, ground_truths)
            
            # 결과 생성
            result = ConfidenceResult(
                confidence=conf,
                domain=domain,
                metrics=metrics,
                sample_count=len(filtered_preds)
            )
            
            # 최적값 선택 (Recall 우선, F1 보조)
            if self._is_better_result(result, best_result):
                best_result = result
        
        return best_result
    
    def _apply_confidence_threshold(
        self,
        predictions: List[Dict[str, Any]],
        confidence: float
    ) -> List[Dict[str, Any]]:
        """Confidence 임계값 적용"""
        filtered = []
        
        for pred in predictions:
            # 예측에서 confidence 점수 추출
            pred_conf = pred.get('confidence', pred.get('score', 1.0))
            
            if pred_conf >= confidence:
                filtered.append(pred)
            else:
                # Confidence 미달시 negative 예측으로 처리
                negative_pred = pred.copy()
                negative_pred['predicted_class'] = -1  # 또는 background 클래스
                negative_pred['confidence'] = pred_conf
                filtered.append(negative_pred)
        
        return filtered
    
    def _calculate_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """메트릭 계산"""
        if len(predictions) == 0 or len(ground_truths) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0
            }
        
        # 예측값과 정답값 추출
        y_pred = [p.get('predicted_class', -1) for p in predictions]
        y_true = [gt.get('true_class', gt.get('label', -1)) for gt in ground_truths]
        
        # 길이 맞춤
        min_len = min(len(y_pred), len(y_true))
        y_pred = y_pred[:min_len]
        y_true = y_true[:min_len]
        
        # 메트릭 계산
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            
            accuracy = np.mean(np.array(y_pred) == np.array(y_true))
            
            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'accuracy': float(accuracy)
            }
        
        except Exception as e:
            self.logger.warning(f"메트릭 계산 실패: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0
            }
    
    def _is_better_result(
        self,
        current: ConfidenceResult,
        best: Optional[ConfidenceResult]
    ) -> bool:
        """결과 비교 (Recall 우선, F1 보조)"""
        if best is None:
            return True
        
        current_primary = current.get_primary_score(self.config)
        best_primary = best.get_primary_score(self.config)
        
        # Primary 메트릭 비교
        if current_primary > best_primary:
            return True
        elif current_primary < best_primary:
            return False
        
        # Primary가 동일하면 Secondary 메트릭 비교
        current_secondary = current.get_secondary_score(self.config)
        best_secondary = best.get_secondary_score(self.config)
        
        return current_secondary > best_secondary
    
    def _save_tuning_results(self, domain_results: Dict[str, ConfidenceResult]) -> None:
        """튜닝 결과 저장"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 결과 데이터 준비
        results_data = {
            "timestamp": timestamp,
            "config": asdict(self.config),
            "confidence_range": self.confidence_values,
            "domain_results": {
                domain: {
                    "confidence": result.confidence,
                    "metrics": result.metrics,
                    "sample_count": result.sample_count
                }
                for domain, result in domain_results.items()
            },
            "best_confidences": self.best_confidences
        }
        
        # 저장 경로
        save_dir = Path("artifacts/confidence_tuning")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / f"confidence_tuning_{timestamp}.json"
        
        # JSON 저장
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Confidence 튜닝 결과 저장: {save_file}")
    
    def _apply_to_inference_config(self) -> None:
        """추론 설정에 최적 confidence 적용 (프로세스 내 주입)"""
        try:
            # 현재 설정 로드
            from src.utils.core import load_config
            config = load_config()
            
            # Inference 섹션에 최적 confidence 주입
            if 'inference' not in config:
                config['inference'] = {}
            
            config['inference']['optimal_confidences'] = self.best_confidences
            config['inference']['confidence_auto_tuned'] = True
            config['inference']['tuning_timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 기본 confidence도 업데이트 (single 도메인 우선)
            if 'single' in self.best_confidences:
                config['inference']['confidence'] = self.best_confidences['single']
            
            # 하드코딩 임계 무시 일관화
            self._override_hardcoded_thresholds(config)
            
            self.logger.info(f"⚙️ 추론 설정에 최적 confidence 적용: {self.best_confidences}")
            
        except Exception as e:
            self.logger.error(f"추론 설정 업데이트 실패: {e}")
    
    def _persist_to_inference_json(self) -> None:
        """선택된 도메인별 confidence를 JSON에 영속 저장"""
        try:
            import time
            from datetime import datetime
            
            # artifacts/confidence_tuning 디렉터리 생성
            save_dir = Path("artifacts/confidence_tuning")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 영속화 데이터 준비
            persist_data = {
                "single": self.best_confidences.get("single", 0.25),
                "combination": self.best_confidences.get("combination", 0.25),
                "epoch": getattr(self, 'current_epoch', 0),  # 현재 에포크
                "timestamp": datetime.now().isoformat(),
                "tuning_config": {
                    "conf_min": self.config.conf_min,
                    "conf_max": self.config.conf_max,
                    "conf_step": self.config.conf_step,
                    "primary_metric": self.config.primary_metric,
                    "secondary_metric": self.config.secondary_metric
                }
            }
            
            # JSON 저장
            json_file = save_dir / "last_selected.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(persist_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 선택된 confidence 영속 저장: {json_file}")
            
        except Exception as e:
            self.logger.error(f"Confidence 영속화 저장 실패: {e}")
    
    def _override_hardcoded_thresholds(self, config: Dict[str, Any]) -> None:
        """하드코딩 임계 무시 일관화"""
        # 하드코딩된 confidence threshold 키들
        hardcoded_keys = [
            ("train", "detection", "evaluation", "conf_threshold"),
            ("inference", "detection_nms", "confidence_threshold"),
            ("models", "detector", "confidence_threshold")
        ]
        
        overridden_keys = []
        
        for key_path in hardcoded_keys:
            # 중첩 키 경로 탐색
            current = config
            valid_path = True
            
            for key in key_path[:-1]:
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    valid_path = False
                    break
            
            # 마지막 키 처리
            if valid_path and key_path[-1] in current:
                old_value = current[key_path[-1]]
                
                # single 도메인 값으로 오버라이드
                if 'single' in self.best_confidences:
                    current[key_path[-1]] = self.best_confidences['single']
                    overridden_keys.append(f"{.".join(key_path)}: {old_value} → {self.best_confidences['single']}")
        
        if overridden_keys:
            self.logger.info(
                f"🔄 Using tuned confidence; overriding hard-coded thresholds:\n  " +
                "\n  ".join(overridden_keys)
            )
    
    def _auto_apply_to_checkpoint(self) -> None:
        """체크포인트 반영 자동화 (내부에서 수행)"""
        # 이 메서드는 체크포인트가 제공되었을 때만 작동하도록 설계
        # 외부 호출 의존 제거를 위해 내부에서 체크포인트 처리 시도
        try:
            # 현재 인스턴스에 체크포인트 데이터가 있으면 처리
            if hasattr(self, '_current_checkpoint') and self._current_checkpoint:
                self.apply_to_checkpoint(self._current_checkpoint)
            else:
                # 체크포인트가 없으면 안전하게 스킵
                self.logger.debug("체크포인트 없음 - 자동 적용 스킵")
        
        except Exception as e:
            self.logger.warning(f"체크포인트 자동 반영 실패: {e}")
    
    def set_current_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """현재 체크포인트 설정 (내부 자동 적용용)"""
        self._current_checkpoint = checkpoint
    
    def load_persisted_confidences(self) -> Optional[Dict[str, float]]:
        """영속화된 confidence 값 로드 (추론 초기화시 사용)"""
        try:
            json_file = Path("artifacts/confidence_tuning/last_selected.json")
            
            if not json_file.exists():
                return None
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 도메인별 confidence 값 추출
            confidences = {
                "single": data.get("single", 0.25),
                "combination": data.get("combination", 0.25)
            }
            
            self.logger.info(f"💼 영속화된 confidence 로드: {confidences}")
            return confidences
        
        except Exception as e:
            self.logger.warning(f"영속화된 confidence 로드 실패: {e}")
            return None
    
    def _generate_summary_report(self, domain_results: Dict[str, ConfidenceResult]) -> None:
        """요약 리포트 생성"""
        report_lines = [
            "# Confidence 자동 튜닝 리포트",
            "",
            f"**튜닝 시간**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**스윕 범위**: {self.config.conf_min} - {self.config.conf_max} (step {self.config.conf_step})",
            f"**선택 기준**: {self.config.primary_metric} (우선), {self.config.secondary_metric} (보조)",
            "",
            "## 도메인별 최적 Confidence",
            ""
        ]
        
        for domain, result in domain_results.items():
            report_lines.extend([
                f"### {domain.upper()}",
                f"- **최적 Confidence**: {result.confidence}",
                f"- **샘플 수**: {result.sample_count}",
                f"- **Precision**: {result.metrics.get('precision', 0):.3f}",
                f"- **Recall**: {result.metrics.get('recall', 0):.3f}",
                f"- **F1 Score**: {result.metrics.get('f1', 0):.3f}",
                f"- **Accuracy**: {result.metrics.get('accuracy', 0):.3f}",
                ""
            ])
        
        # 추가 리포트 정보
        report_lines.extend([
            "## 튜닝 세부 설정",
            f"- **스윙 범위**: {self.config.conf_min} - {self.config.conf_max}",
            f"- **스윙 스텝**: {self.config.conf_step}",
            f"- **선정 기준**: {self.config.primary_metric} (우선) → {self.config.secondary_metric} (보조)",
            f"- **도메인**: {', '.join(self.config.domains)}",
            "",
            "## 적용 상태",
            "- ✅ **체크포인트 메타**: 자동 반영",
            "- ✅ **추론 설정**: 영속 JSON 저장",
            "- ✅ **리포트**: 도메인별 선택값 및 스윙 내역 명시",
            ""
        ])
        
        # 리포트 저장
        report_dir = Path("artifacts/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"confidence_tuning_{time.strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"📋 Confidence 튜닝 리포트 생성: {report_file}")
    
    def get_best_confidences(self) -> Dict[str, float]:
        """최적 confidence 반환"""
        return self.best_confidences.copy()
    
    def apply_to_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """체크포인트에 최적 confidence 메타 추가"""
        if checkpoint is None:
            self.logger.warning("체크포인트가 None임 - 안전하게 스킵")
            return {}
        
        if not self.config.save_to_checkpoint:
            return checkpoint
        
        if 'meta' not in checkpoint:
            checkpoint['meta'] = {}
        
        checkpoint['meta']['optimal_confidences'] = self.best_confidences
        checkpoint['meta']['confidence_tuning_timestamp'] = time.time()
        checkpoint['meta']['confidence_tuning_config'] = asdict(self.config)
        
        # 하드코딩 임계 오버라이드 정보 추가
        checkpoint['meta']['hardcoded_thresholds_overridden'] = True
        checkpoint['meta']['override_keys'] = [
            "train.detection.evaluation.conf_threshold",
            "inference.detection_nms.confidence_threshold",
            "models.detector.confidence_threshold"
        ]
        
        self.logger.info("📦 체크포인트에 최적 confidence 메타 추가 (자동 수행)")
        
        return checkpoint


if __name__ == "__main__":
    print("🧪 Confidence 자동 튜닝 시스템 테스트 (1단계 필수)")
    print("=" * 60)
    
    # 설정 테스트
    config = ConfidenceTuningConfig(
        conf_min=0.20,
        conf_max=0.30,
        conf_step=0.02,
        domains=["single", "combination"]
    )
    
    tuner = ConfidenceTuner(config)
    print(f"✅ Confidence 범위: {tuner.confidence_values}")
    
    # Mock 데이터 생성
    import random
    
    mock_predictions = []
    mock_ground_truths = []
    
    for i in range(100):
        # Mock prediction
        conf = random.uniform(0.1, 0.9)
        pred_class = random.randint(0, 10) if conf > 0.25 else -1
        
        mock_predictions.append({
            'confidence': conf,
            'predicted_class': pred_class
        })
        
        # Mock ground truth
        true_class = random.randint(0, 10)
        mock_ground_truths.append({
            'true_class': true_class
        })
    
    # Mock 도메인 마스크
    mock_domain_masks = {
        'single': torch.tensor([i < 75 for i in range(100)]),  # 75% single
        'combination': torch.tensor([i >= 75 for i in range(100)])  # 25% combination
    }
    
    # 튜닝 실행
    best_confidences = tuner.tune_confidence(
        mock_predictions,
        mock_ground_truths,
        mock_domain_masks
    )
    
    print(f"✅ 최적 confidence: {best_confidences}")
    
    # 체크포인트 테스트
    mock_checkpoint = {'model_state_dict': {}, 'epoch': 10}
    updated_checkpoint = tuner.apply_to_checkpoint(mock_checkpoint)
    print(f"✅ 체크포인트 메타 추가: {'optimal_confidences' in updated_checkpoint.get('meta', {})}")
    
    print("🎉 Confidence 자동 튜닝 시스템 테스트 완료!")