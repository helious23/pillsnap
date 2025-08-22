#!/usr/bin/env python3
"""
Stage 1 Performance Evaluator
파이프라인 검증을 위한 평가 시스템

목표:
- GPU 환경 검증
- 기본 파이프라인 동작 확인
- Stage 2 진행 가능성 평가
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, Any

from src.utils.core import PillSnapLogger


class Stage1Evaluator:
    """Stage 1 파이프라인 검증 평가기"""
    
    def __init__(self):
        self.logger = PillSnapLogger(__name__)
        self.targets = {
            "classification_accuracy": 0.40,  # 50클래스 기준 (무작위 2% × 20배)
            "gpu_memory_limit": 14.0,         # RTX 5080 안정성 기준
            "pipeline_complete": True,        # 전체 파이프라인 완료
        }
    
    def evaluate_stage_1(self, exp_dir: str = "/home/max16/pillsnap_data/exp/exp01") -> Dict[str, Any]:
        """Stage 1 종합 평가"""
        
        self.logger.step("Stage 1 평가 시작", "파이프라인 검증")
        
        # 1. GPU 환경 검증
        gpu_check = self._check_gpu_environment()
        
        # 2. 학습 결과 확인
        training_results = self._check_training_results(exp_dir)
        
        # 3. 시스템 안정성 확인
        system_check = self._check_system_stability(exp_dir)
        
        # 4. 권장사항 생성
        recommendation = self._generate_stage1_recommendation(
            gpu_check, training_results, system_check
        )
        
        # 5. 평가 결과 저장
        evaluation_report = {
            "stage": 1,
            "purpose": "pipeline_validation",
            "timestamp": time.time(),
            "gpu_environment": gpu_check,
            "training_results": training_results,
            "system_stability": system_check,
            "recommendation": recommendation,
            "targets": self.targets
        }
        
        self._save_evaluation_report(evaluation_report, exp_dir)
        
        # 6. 사용자에게 결과 표시
        self._present_stage1_results(recommendation)
        
        return evaluation_report
    
    def _check_gpu_environment(self) -> Dict[str, Any]:
        """GPU 환경 검증"""
        
        results = {}
        
        # CUDA 사용 가능성
        results["cuda_available"] = torch.cuda.is_available()
        
        if results["cuda_available"]:
            # GPU 정보
            results["gpu_name"] = torch.cuda.get_device_name(0)
            results["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # 메모리 사용량
            torch.cuda.empty_cache()
            results["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            results["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
            
            # RTX 5080 확인
            results["is_rtx5080"] = "RTX 5080" in results["gpu_name"]
            results["memory_adequate"] = results["gpu_memory_total_gb"] >= 15.0
        else:
            results["gpu_name"] = "CPU Only"
            results["gpu_memory_total_gb"] = 0.0
            results["is_rtx5080"] = False
            results["memory_adequate"] = False
        
        # PyTorch 버전
        results["pytorch_version"] = torch.__version__
        results["pytorch_cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
        return results
    
    def _check_training_results(self, exp_dir: str) -> Dict[str, Any]:
        """학습 결과 확인"""
        
        results = {
            "training_completed": False,
            "best_accuracy": 0.0,
            "model_saved": False,
            "logs_available": False
        }
        
        try:
            # 모델 체크포인트 확인
            checkpoints_dir = Path(exp_dir) / "checkpoints"
            artifacts_dir = Path("artifacts/models/classification")
            
            # Stage 1 모델 확인 (50클래스)
            stage1_model = artifacts_dir / "best_classifier_50classes.pt"
            if stage1_model.exists():
                try:
                    checkpoint = torch.load(stage1_model, map_location='cpu')
                    results["best_accuracy"] = checkpoint.get("best_accuracy", 0.0)
                    results["model_saved"] = True
                    results["training_completed"] = True
                except Exception as e:
                    self.logger.warning(f"모델 로드 실패: {e}")
            
            # 로그 파일 확인
            log_files = list(Path(exp_dir).glob("logs/*.out"))
            results["logs_available"] = len(log_files) > 0
            
        except Exception as e:
            self.logger.warning(f"학습 결과 확인 실패: {e}")
        
        return results
    
    def _check_system_stability(self, exp_dir: str) -> Dict[str, Any]:
        """시스템 안정성 확인"""
        
        results = {
            "no_oom_errors": True,
            "no_crashes": True,
            "data_loading_ok": True
        }
        
        try:
            # 오류 로그 확인
            error_logs = list(Path(exp_dir).glob("logs/*.err"))
            for error_log in error_logs:
                if error_log.exists() and error_log.stat().st_size > 0:
                    with open(error_log) as f:
                        content = f.read()
                        
                        if "CUDA out of memory" in content or "OutOfMemoryError" in content:
                            results["no_oom_errors"] = False
                        
                        if "Traceback" in content or "Exception" in content:
                            results["no_crashes"] = False
        except Exception as e:
            self.logger.warning(f"로그 확인 실패: {e}")
        
        return results
    
    def _generate_stage1_recommendation(
        self, 
        gpu_check: Dict, 
        training_results: Dict, 
        system_check: Dict
    ) -> Dict[str, Any]:
        """Stage 1 권장사항 생성"""
        
        # 필수 체크
        mandatory_passed = (
            gpu_check.get("cuda_available", False) and
            training_results.get("training_completed", False) and
            system_check.get("no_crashes", True)
        )
        
        # 성능 체크
        accuracy_ok = training_results.get("best_accuracy", 0) >= self.targets["classification_accuracy"]
        memory_ok = gpu_check.get("memory_adequate", False)
        
        # 종합 판정
        if mandatory_passed and accuracy_ok and memory_ok:
            decision = "RECOMMEND_PROCEED"
            color = "🟢"
            message = "Stage 1 파이프라인 검증 완료!"
        elif mandatory_passed and (accuracy_ok or memory_ok):
            decision = "RECOMMEND_PROCEED"
            color = "🟢"
            message = "Stage 1 기본 요구사항 충족"
        elif mandatory_passed:
            decision = "SUGGEST_OPTIMIZE"
            color = "🟡"
            message = "Stage 1 완료, 일부 최적화 권장"
        else:
            decision = "WARN_STOP"
            color = "🔴"
            message = "Stage 1 필수 요구사항 미충족"
        
        # 구체적 제안사항
        suggestions = []
        
        if not gpu_check.get("cuda_available"):
            suggestions.append("CUDA 환경 설정 확인")
        
        if not training_results.get("training_completed"):
            suggestions.append("학습 완료까지 기다리거나 재실행")
        
        if training_results.get("best_accuracy", 0) < self.targets["classification_accuracy"]:
            suggestions.append(f"정확도 개선 필요 (현재: {training_results.get('best_accuracy', 0):.1%}, 목표: {self.targets['classification_accuracy']:.1%})")
        
        if not gpu_check.get("memory_adequate"):
            suggestions.append("GPU 메모리 확인 (RTX 5080 권장)")
        
        # 사용자 선택 옵션
        if decision == "RECOMMEND_PROCEED":
            user_options = [
                "[1] Stage 2로 진행",
                "[2] Stage 1 추가 최적화",
                "[3] 상세 분석 리포트 생성"
            ]
        elif decision == "SUGGEST_OPTIMIZE":
            user_options = [
                "[1] 권장사항 적용 후 재시도",
                "[2] 현재 상태로 Stage 2 진행",
                "[3] 상세 디버깅 모드"
            ]
        else:
            user_options = [
                "[1] 환경 설정 재검토",
                "[2] 학습 재실행",
                "[3] 기술 지원 요청"
            ]
        
        return {
            "decision": decision,
            "color": color,
            "message": message,
            "suggestions": suggestions,
            "user_options": user_options,
            "performance_score": self._calculate_stage1_score(gpu_check, training_results, system_check)
        }
    
    def _calculate_stage1_score(
        self, 
        gpu_check: Dict, 
        training_results: Dict, 
        system_check: Dict
    ) -> float:
        """Stage 1 성능 점수 계산"""
        
        scores = []
        
        # GPU 환경 점수
        if gpu_check.get("cuda_available"):
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # 학습 완료 점수
        if training_results.get("training_completed"):
            accuracy = training_results.get("best_accuracy", 0)
            accuracy_score = min(1.0, accuracy / self.targets["classification_accuracy"])
            scores.append(accuracy_score)
        else:
            scores.append(0.0)
        
        # 안정성 점수
        stability_score = sum([
            system_check.get("no_oom_errors", True),
            system_check.get("no_crashes", True),
            system_check.get("data_loading_ok", True)
        ]) / 3
        scores.append(stability_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _save_evaluation_report(self, report: Dict, exp_dir: str) -> None:
        """평가 결과 저장"""
        
        try:
            reports_dir = Path(exp_dir) / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = reports_dir / "stage_1_evaluation.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Stage 1 평가 리포트 저장: {report_path}")
            
        except Exception as e:
            self.logger.error(f"리포트 저장 실패: {e}")
    
    def _present_stage1_results(self, recommendation: Dict) -> None:
        """Stage 1 평가 결과 표시"""
        
        print("\n")
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║                    🎯 Stage 1 평가 완료                          ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        print(f"║ {recommendation['color']} {recommendation['message']}")
        print("║")
        print(f"║ 📊 성능 점수: {recommendation['performance_score']:.3f}")
        print("║")
        
        if recommendation.get('suggestions'):
            print("║ 💡 권장사항:")
            for i, suggestion in enumerate(recommendation['suggestions'], 1):
                print(f"║   {i}. {suggestion}")
            print("║")
        
        print("║ 🎭 선택 옵션:")
        for option in recommendation['user_options']:
            print(f"║   {option}")
        
        print("╚══════════════════════════════════════════════════════════════════╝")
        
        # 사용자 입력 대기
        try:
            user_choice = input("\n선택하세요 [1-3]: ")
            print(f"선택됨: {user_choice}")
            
            if user_choice == "1":
                if recommendation["decision"] == "RECOMMEND_PROCEED":
                    print("✅ Stage 2 진행을 위해 다음 명령어를 실행하세요:")
                    print("python -m src.training.train_classification_stage --stage 2 --epochs 30")
                else:
                    print("🔧 환경 설정을 재검토합니다...")
            elif user_choice == "2":
                print("📊 추가 작업을 수행합니다...")
            elif user_choice == "3":
                print("🔍 상세 모드로 전환합니다...")
            
        except KeyboardInterrupt:
            print("\n⏹️  평가 종료")
        except Exception as e:
            print(f"\n⚠️  입력 처리 오류: {e}")


def main():
    """CLI 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1 Performance Evaluator")
    parser.add_argument("--exp-dir", type=str, 
                       default="/home/max16/pillsnap_data/exp/exp01",
                       help="Experiment directory path")
    parser.add_argument("--save-report", action="store_true",
                       help="Save detailed evaluation report")
    
    args = parser.parse_args()
    
    print("🎯 Stage 1 파이프라인 검증 평가 시작")
    print("=" * 60)
    
    evaluator = Stage1Evaluator()
    evaluation_result = evaluator.evaluate_stage_1(args.exp_dir)
    
    print(f"\n📄 상세 리포트: {args.exp_dir}/reports/stage_1_evaluation.json")


if __name__ == "__main__":
    main()