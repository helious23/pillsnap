"""
PillSnap ML Stage 1 Training Launcher
Stage 1 학습 시스템 런처

새로운 구조:
- src/training/train_interleaved_pipeline.py 호출
- 상업용 수준의 체계적 학습 관리
- 완전한 목표 검증 시스템
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train_interleaved_pipeline import main as training_main


if __name__ == "__main__":
    print("🚀 PillSnap ML Stage 1 Training System")
    print("상세한 학습 파이프라인 시작...")
    print("=" * 60)
    
    try:
        # 새로운 체계적 학습 시스템 실행
        training_main()
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 학습 시스템 오류: {e}")
        import traceback
        traceback.print_exc()
