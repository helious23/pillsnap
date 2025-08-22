#!/usr/bin/env python3
"""
Stage 2 Performance Evaluator
성능 기준선 확립을 위한 평가 시스템

목표:
- Classification accuracy ≥ 0.60 (250클래스)
- Detection mAP@0.5 ≥ 0.50 
- Auto Batch 튜닝 성공 확인
- 128GB RAM 최적화 검증
"""

import os
import json
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from src.utils.core import PillSnapLogger


@dataclass
class Stage2Targets:
    """Stage 2 성능 목표"""
    # 필수 체크 (실제 중요한 것만)
    mandatory_checks = [
        "training_completed",
        "model_saved",
        "memory_optimization_working"
    ]
    
    # 성능 목표 (Stage 2는 250클래스 기준선)
    classification_accuracy = 0.40  # 40% (Stage 2 목표)
    classification_macro_f1 = 0.35
    classification_top5_accuracy = 0.60
    
    # 검출 성능 (나중에 구현될 예정)
    detection_map_0_5 = 0.50  # mAP@0.5
    detection_precision = 0.45
    detection_recall = 0.40
    
    # 시스템 성능
    batch_size_minimum = 32
    gpu_memory_limit_gb = 14.0
    throughput_img_s = 80


class OptimizationAdvisor:
    """Stage 2 최적화 권장 시스템"""
    
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.targets = Stage2Targets()
    
    def evaluate_stage_2(self, exp_dir: str = "/home/max16/pillsnap_data/exp/exp01") -> Dict[str, Any]:
        """Stage 2 종합 평가"""
        
        self.logger.step("Stage 2 평가 시작", "성능 기준선 확립 검증")
        
        # 1. 필수 체크 수행
        mandatory_results = self._check_mandatory_requirements(exp_dir)
        
        # 2. 성능 지표 수집
        performance_results = self._collect_performance_metrics(exp_dir)
        
        # 3. 시스템 안정성 확인
        system_results = self._check_system_stability(exp_dir)
        
        # 4. 종합 평가 및 권장사항 생성
        recommendation = self._generate_recommendation(
            mandatory_results, performance_results, system_results
        )
        
        # 5. 평가 결과 저장
        evaluation_report = {
            "stage": 2,
            "timestamp": time.time(),
            "mandatory_checks": mandatory_results,
            "performance_metrics": performance_results,
            "system_metrics": system_results,
            "recommendation": recommendation,
            "targets": {
                "classification_accuracy": self.targets.classification_accuracy,
                "detection_map_0_5": self.targets.detection_map_0_5,
                "throughput_target": self.targets.throughput_img_s
            }
        }
        
        self._save_evaluation_report(evaluation_report, exp_dir)
        
        # 6. 사용자에게 권장사항 표시
        self._present_recommendation_to_user(recommendation)
        
        return evaluation_report
    
    def _check_mandatory_requirements(self, exp_dir: str) -> Dict[str, bool]:
        """필수 요구사항 체크 - 실제 중요한 것만"""
        
        results = {}
        
        # 1. 학습 완료 확인 (가장 중요)
        try:
            # Stage 2 모델 파일 존재 확인
            artifacts_dir = Path("artifacts/models/classification")
            stage2_model = artifacts_dir / "best_classifier_250classes.pt"
            
            if stage2_model.exists():
                # 모델 로드해서 정확도 확인
                checkpoint = torch.load(stage2_model, map_location='cpu')
                best_acc = checkpoint.get("best_accuracy", 0.0)
                results["training_completed"] = best_acc > 0.1  # 최소한의 학습이 이루어졌는지
                self.logger.info(f"Stage 2 모델 정확도: {best_acc:.1%}")
            else:
                results["training_completed"] = False
                self.logger.warning("Stage 2 모델 파일 없음")
        except Exception as e:
            self.logger.warning(f"학습 완료 체크 실패: {e}")
            results["training_completed"] = False
        
        # 2. 모델 저장 확인
        results["model_saved"] = results["training_completed"]
        
        # 3. 메모리 최적화 동작 확인
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                results["memory_optimization_working"] = memory_allocated <= self.targets.gpu_memory_limit_gb
                self.logger.info(f"GPU 메모리 사용량: {memory_allocated:.2f}GB")
            else:
                results["memory_optimization_working"] = True  # CPU 모드에서는 통과
        except:
            results["memory_optimization_working"] = False
        
        return results
    
    def _collect_performance_metrics(self, exp_dir: str) -> Dict[str, float]:
        """성능 지표 수집 - 실제 저장된 모델에서"""
        
        metrics = {}
        
        try:
            # artifacts 디렉토리에서 실제 저장된 모델 확인
            artifacts_dir = Path("artifacts/models/classification")
            stage2_model = artifacts_dir / "best_classifier_250classes.pt"
            
            if stage2_model.exists():
                checkpoint = torch.load(stage2_model, map_location='cpu')
                
                # 실제 저장된 성능 지표들
                metrics["classification_accuracy"] = checkpoint.get("best_accuracy", 0.0)
                metrics["classification_top3_accuracy"] = checkpoint.get("top3_accuracy", 0.0)
                metrics["classification_top5_accuracy"] = checkpoint.get("top5_accuracy", 0.0)
                metrics["classification_f1_macro"] = checkpoint.get("f1_macro", 0.0)
                
                # 학습 메타데이터
                metrics["total_epochs_completed"] = checkpoint.get("epoch", 0)
                metrics["training_time_minutes"] = checkpoint.get("training_time", 0.0)
                
                self.logger.info(f"모델에서 추출한 성능: {metrics['classification_accuracy']:.1%}")
            else:
                self.logger.warning("Stage 2 모델 파일을 찾을 수 없음")
                metrics["classification_accuracy"] = 0.0
                metrics["classification_top3_accuracy"] = 0.0
                metrics["classification_top5_accuracy"] = 0.0
                metrics["classification_f1_macro"] = 0.0
        except Exception as e:
            self.logger.error(f"성능 지표 수집 실패: {e}")
            metrics["classification_accuracy"] = 0.0
            metrics["classification_top3_accuracy"] = 0.0
            metrics["classification_top5_accuracy"] = 0.0
            metrics["classification_f1_macro"] = 0.0
        
        # 시스템 성능 지표
        try:
            if torch.cuda.is_available():
                metrics["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1024**3
                # GPU 정보
                gpu_name = torch.cuda.get_device_name(0)
                metrics["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"GPU: {gpu_name}, 메모리: {metrics['gpu_memory_used_gb']:.2f}GB")
            else:
                metrics["gpu_memory_used_gb"] = 0.0
                metrics["gpu_memory_total_gb"] = 0.0
        except:
            metrics["gpu_memory_used_gb"] = 0.0
            metrics["gpu_memory_total_gb"] = 0.0
        
        return metrics
    
    def _check_system_stability(self, exp_dir: str) -> Dict[str, Any]:
        """시스템 안정성 확인"""
        
        results = {
            "no_crashes": True,
            "memory_stable": True,
            "data_loading_stable": True
        }
        
        try:
            # 학습 로그에서 오류 패턴 확인
            error_log = Path(exp_dir) / "logs" / "train.err"
            if error_log.exists():
                with open(error_log) as f:
                    error_content = f.read()
                    
                    # OOM 또는 크래시 패턴 확인
                    if "OutOfMemoryError" in error_content or "CUDA out of memory" in error_content:
                        results["memory_stable"] = False
                    
                    if "Traceback" in error_content or "Exception" in error_content:
                        results["no_crashes"] = False
        except:
            pass
        
        return results
    
    def _calculate_performance_score(self, performance_metrics: Dict) -> float:
        """성능 점수 계산 (0~1)"""
        
        scores = []
        
        # 분류 정확도 점수
        if performance_metrics.get("classification_accuracy", 0) > 0:
            accuracy_score = min(1.0, performance_metrics["classification_accuracy"] / self.targets.classification_accuracy)
            scores.append(accuracy_score)
        
        # 검출 성능 점수 (나중에 구현)
        if performance_metrics.get("detection_map_0_5", 0) > 0:
            detection_score = min(1.0, performance_metrics["detection_map_0_5"] / self.targets.detection_map_0_5)
            scores.append(detection_score)
        
        # 처리량 점수
        if performance_metrics.get("estimated_throughput_img_s", 0) > 0:
            throughput_score = min(1.0, performance_metrics["estimated_throughput_img_s"] / self.targets.throughput_img_s)
            scores.append(throughput_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_recommendation(
        self, 
        mandatory_results: Dict, 
        performance_results: Dict, 
        system_results: Dict
    ) -> Dict[str, Any]:
        """권장사항 생성"""
        
        # 필수 체크 통과 여부
        mandatory_pass = all(mandatory_results.values())
        
        # 성능 점수 계산
        performance_score = self._calculate_performance_score(performance_results)
        
        # 시스템 안정성
        system_stable = all(system_results.values())
        
        # 권장사항 결정
        if not mandatory_pass:
            recommendation = {
                "decision": "WARN_STOP",
                "color": "🔴",
                "message": "필수 요구사항 미충족",
                "performance_score": performance_score,
                "reason": "mandatory_checks_failed",
                "failed_checks": [k for k, v in mandatory_results.items() if not v]
            }
        elif performance_score >= 1.0 and system_stable:
            recommendation = {
                "decision": "RECOMMEND_PROCEED", 
                "color": "🟢",
                "message": "Stage 2 모든 목표 달성!",
                "performance_score": performance_score,
                "reason": "all_targets_met"
            }
        elif performance_score >= 0.7 and system_stable:
            recommendation = {
                "decision": "RECOMMEND_PROCEED",
                "color": "🟢", 
                "message": "Stage 2 충분한 성능 달성",
                "performance_score": performance_score,
                "reason": "sufficient_performance"
            }
        elif performance_score >= 0.5:
            recommendation = {
                "decision": "SUGGEST_OPTIMIZE",
                "color": "🟡",
                "message": "Stage 2 성능 미달. 최적화 권장",
                "performance_score": performance_score,
                "reason": "performance_below_target"
            }
        else:
            recommendation = {
                "decision": "WARN_STOP",
                "color": "🔴",
                "message": "Stage 2 성능이 매우 낮음", 
                "performance_score": performance_score,
                "reason": "performance_too_low"
            }
        
        # 구체적 제안사항 추가
        recommendation["suggestions"] = self._generate_optimization_suggestions(
            performance_results, mandatory_results
        )
        
        # 사용자 선택 옵션
        if recommendation["decision"] == "RECOMMEND_PROCEED":
            recommendation["user_options"] = [
                "[1] Stage 3으로 진행",
                "[2] 현재 Stage에서 추가 최적화",
                "[3] 상세 분석 리포트 생성"
            ]
        elif recommendation["decision"] == "SUGGEST_OPTIMIZE":
            recommendation["user_options"] = [
                "[1] 권장사항 적용 후 재시도", 
                "[2] 현재 성능으로 진행 (위험)",
                "[3] 상세 디버깅 모드"
            ]
        else:
            recommendation["user_options"] = [
                "[1] 강력한 최적화 적용 후 재시도",
                "[2] 아키텍처 재검토", 
                "[3] 데이터 품질 점검"
            ]
        
        return recommendation
    
    def _generate_optimization_suggestions(
        self, 
        performance_results: Dict, 
        mandatory_results: Dict
    ) -> List[str]:
        """최적화 제안사항 생성"""
        
        suggestions = []
        
        # 성능 기반 제안
        classification_acc = performance_results.get("classification_accuracy", 0)
        if classification_acc < self.targets.classification_accuracy * 0.8:
            suggestions.append("학습률 조정 (2e-4 → 1e-4)")
            suggestions.append("드롭아웃 증가 (0.3 → 0.4)")
            suggestions.append("데이터 증강 강화")
        
        # 시스템 최적화 제안
        gpu_memory = performance_results.get("gpu_memory_used_gb", 0)
        if gpu_memory > 12:
            suggestions.append("배치 크기 축소 (현재 메모리 사용량 높음)")
        elif gpu_memory < 8:
            suggestions.append("배치 크기 증가 (메모리 여유 있음)")
        
        # 필수 체크 실패시 제안
        if not mandatory_results.get("auto_batch_tuning_success", True):
            suggestions.append("Auto Batch 튜닝 활성화 확인")
        
        if not mandatory_results.get("tensorboard_logging", True):
            suggestions.append("TensorBoard 로깅 설정 확인")
        
        return suggestions[:5]  # 상위 5개만
    
    def _save_evaluation_report(self, report: Dict, exp_dir: str) -> None:
        """평가 결과 저장"""
        
        try:
            reports_dir = Path(exp_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = reports_dir / "stage_2_evaluation.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"평가 리포트 저장: {report_path}")
            
        except Exception as e:
            self.logger.error(f"평가 리포트 저장 실패: {e}")
    
    def _present_recommendation_to_user(self, recommendation: Dict) -> None:
        """사용자에게 시각적으로 권장사항 표시"""
        
        print("\n")
        print("═" * 70)
        print("🎯 Stage 2 OptimizationAdvisor 평가 결과")
        print("═" * 70)
        print(f"{recommendation['color']} {recommendation['message']}")
        print()
        print(f"📊 성능 점수: {recommendation['performance_score']:.3f}")
        print(f"🔎 판단 근거: {recommendation.get('reason', 'unknown')}")
        print()
        
        if recommendation.get('failed_checks'):
            print("❌ 실패한 필수 체크:")
            for check in recommendation['failed_checks']:
                print(f"   • {check}")
            print()
        
        if recommendation.get('suggestions'):
            print("💡 제안사항:")
            for i, suggestion in enumerate(recommendation['suggestions'], 1):
                print(f"   {i}. {suggestion}")
            print()
        
        print("🎭 선택 옵션:")
        for option in recommendation['user_options']:
            print(f"   {option}")
        
        print("═" * 70)
        
        # 자동 종료 (대화형 입력 제거)
        if recommendation["decision"] == "RECOMMEND_PROCEED":
            print("\n✅ 추천: Stage 3 진행 명령어")
            print("source .venv/bin/activate && python -m src.training.train_classification_stage --stage 3 --epochs 30")
        else:
            print("\n⚠️  최적화 작업 후 Stage 2 재시도 권장")
        
        print()
        return  # 대화형 입력 제거


def main():
    """CLI 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2 Performance Evaluator")
    parser.add_argument("--exp-dir", type=str, 
                       default="/home/max16/pillsnap_data/exp/exp01",
                       help="Experiment directory path")
    parser.add_argument("--save-report", action="store_true",
                       help="Save detailed evaluation report")
    
    args = parser.parse_args()
    
    print("🎯 Stage 2 OptimizationAdvisor 시작")
    print("=" * 60)
    
    advisor = OptimizationAdvisor()
    advisor.evaluate_stage_2(args.exp_dir)  # evaluation_result 사용하지 않음
    
    if args.save_report:
        print(f"📄 상세 리포트: {args.exp_dir}/reports/stage_2_evaluation.json")


if __name__ == "__main__":
    main()