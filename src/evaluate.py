"""
PillSnap ML Stage 1 Evaluation Launcher
Stage 1 평가 시스템 런처

새로운 구조:
- src/evaluation/evaluate_stage1_targets.py 호출
- 완전한 목표 달성 검증 시스템
- 상업용 수준의 체계적 평가 관리
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate_stage1_targets import main as evaluation_main


if __name__ == "__main__":
    print("📊 PillSnap ML Stage 1 Evaluation System")
    print("상세한 목표 달성 검증 시작...")
    print("=" * 60)
    
    try:
        # 새로운 체계적 평가 시스템 실행
        evaluation_main()
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 평가가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 평가 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
