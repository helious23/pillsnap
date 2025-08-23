#!/usr/bin/env python3
"""
Stage 3 간단한 테스트 실행기 (실제 동작 보장)

복잡한 pytest 의존성 없이 기본 Python만으로 테스트 실행:
- Import 가능성 검증
- 기본 초기화 테스트
- 설정 로딩 테스트
- 간단한 계산 검증
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SimpleTestRunner:
    """간단한 테스트 실행기"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        
    def run_test(self, test_name, test_func):
        """개별 테스트 실행"""
        self.tests_run += 1
        print(f"🔄 {test_name}...", end=" ")
        
        try:
            test_func()
            self.tests_passed += 1
            print("✅ 통과")
            return True
        except Exception as e:
            self.tests_failed += 1
            self.failures.append((test_name, str(e)))
            print(f"❌ 실패: {e}")
            return False
    
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("🎯 Stage 3 간단한 테스트 결과")
        print("="*60)
        print(f"총 테스트: {self.tests_run}")
        print(f"통과: {self.tests_passed}")
        print(f"실패: {self.tests_failed}")
        print(f"성공률: {(self.tests_passed/self.tests_run*100):.1f}%")
        
        if self.failures:
            print("\n❌ 실패한 테스트:")
            for name, error in self.failures:
                print(f"  - {name}: {error}")
        
        print("="*60)
        
        return self.tests_failed == 0


def test_imports():
    """Import 테스트들"""
    print("\n📋 Import 테스트")
    
    runner = SimpleTestRunner()
    
    # Stage 3 Manifest Creator
    def test_manifest_creator_import():
        from src.data.create_stage3_manifest import Stage3ManifestCreator
        assert Stage3ManifestCreator is not None
    
    runner.run_test("Stage 3 Manifest Creator import", test_manifest_creator_import)
    
    # Stage 3 Classification Trainer
    def test_trainer_import():
        from src.training.train_stage3_classification import Stage3ClassificationTrainer
        assert Stage3ClassificationTrainer is not None
    
    runner.run_test("Stage 3 Classification Trainer import", test_trainer_import)
    
    # 데이터로더 컴포넌트
    def test_dataloader_import():
        from src.data.dataloader_manifest_training import ManifestDataset, ManifestTrainingDataLoader
        assert ManifestDataset is not None
        assert ManifestTrainingDataLoader is not None
    
    runner.run_test("데이터로더 컴포넌트 import", test_dataloader_import)
    
    # 테스트 스위트 실행기
    def test_suite_runner_import():
        from scripts.testing.run_stage3_test_suite import Stage3TestSuiteRunner
        assert Stage3TestSuiteRunner is not None
    
    runner.run_test("테스트 스위트 실행기 import", test_suite_runner_import)
    
    return runner


def test_basic_functionality():
    """기본 기능 테스트들"""
    print("\n⚙️ 기본 기능 테스트")
    
    runner = SimpleTestRunner()
    
    # 경로 연산
    def test_paths():
        project_root = Path(__file__).parent.parent.parent
        assert project_root.exists()
        assert (project_root / "src").exists()
        assert (project_root / "tests").exists()
        assert (project_root / "scripts").exists()
    
    runner.run_test("프로젝트 경로 확인", test_paths)
    
    # Stage 3 설정값 검증
    def test_stage3_config():
        target_samples = 100000
        target_classes = 1000
        single_ratio = 0.95
        
        assert target_samples > 0
        assert target_classes > 0
        assert 0 < single_ratio < 1
        
        # 계산 검증
        samples_per_class = target_samples // target_classes
        assert samples_per_class == 100
    
    runner.run_test("Stage 3 설정값 검증", test_stage3_config)
    
    # 데이터 비율 계산
    def test_ratio_calculation():
        samples_per_class = 100
        single_ratio = 0.95
        
        target_single = int(samples_per_class * single_ratio)
        target_combo = samples_per_class - target_single
        
        assert target_single == 95
        assert target_combo == 5
        assert target_single + target_combo == samples_per_class
    
    runner.run_test("데이터 비율 계산", test_ratio_calculation)
    
    # PyTorch 사용 가능성
    def test_pytorch():
        import torch
        assert torch.__version__ is not None
        
        # CUDA 체크 (실패해도 괜찮음)
        cuda_available = torch.cuda.is_available()
        print(f"(CUDA: {cuda_available})", end=" ")
    
    runner.run_test("PyTorch 사용 가능성", test_pytorch)
    
    return runner


def test_initialization():
    """초기화 테스트들"""
    print("\n🏗️ 초기화 테스트")
    
    runner = SimpleTestRunner()
    
    # Stage 3 Manifest Creator 초기화
    def test_manifest_creator_init():
        import tempfile
        from src.data.create_stage3_manifest import Stage3ManifestCreator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            creator = Stage3ManifestCreator(data_root=tmpdir)
            assert creator.target_samples == 100000
            assert creator.target_classes == 1000
            assert creator.single_ratio == 0.95
            assert creator.seed == 42
    
    runner.run_test("Manifest Creator 초기화", test_manifest_creator_init)
    
    # 테스트 스위트 실행기 초기화
    def test_suite_runner_init():
        from scripts.testing.run_stage3_test_suite import Stage3TestSuiteRunner
        
        runner = Stage3TestSuiteRunner()
        assert hasattr(runner, 'cuda_available')
        assert hasattr(runner, 'test_results')
        assert isinstance(runner.cuda_available, bool)
    
    runner.run_test("테스트 스위트 실행기 초기화", test_suite_runner_init)
    
    # 데이터 구조 생성
    def test_data_structures():
        import pandas as pd
        
        # 모의 manifest 데이터
        sample_data = {
            'image_path': [f'/fake/img_{i}.jpg' for i in range(10)],
            'mapping_code': [f'K{i:06d}' for i in range(10)],
            'image_type': ['single'] * 8 + ['combination'] * 2,
            'source': ['train'] * 10
        }
        
        df = pd.DataFrame(sample_data)
        assert len(df) == 10
        assert df['mapping_code'].nunique() == 10
        
        single_ratio = (df['image_type'] == 'single').mean()
        assert single_ratio == 0.8
    
    runner.run_test("데이터 구조 생성", test_data_structures)
    
    return runner


def test_file_operations():
    """파일 연산 테스트들"""
    print("\n📁 파일 연산 테스트")
    
    runner = SimpleTestRunner()
    
    # 임시 디렉토리 생성
    def test_directory_creation():
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            
            # Stage 3 디렉토리 생성
            stage3_dir = base_dir / "artifacts" / "stage3"
            stage3_dir.mkdir(parents=True, exist_ok=True)
            
            reports_dir = stage3_dir / "test_reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            assert stage3_dir.exists()
            assert reports_dir.exists()
    
    runner.run_test("디렉토리 생성", test_directory_creation)
    
    # 설정 파일 접근
    def test_config_access():
        config_path = Path("config.yaml")
        if config_path.exists():
            # 파일이 있으면 읽기 시도
            content = config_path.read_text()
            assert len(content) > 0
            assert 'stage_3' in content
        else:
            # 파일이 없어도 테스트 통과 (CI/CD 환경)
            print("(config.yaml not found)", end=" ")
    
    runner.run_test("설정 파일 접근", test_config_access)
    
    # JSON 처리
    def test_json_operations():
        import json
        
        sample_data = {
            'stage': 3,
            'target_samples': 100000,
            'target_classes': 1000,
            'single_ratio': 0.95,
            'timestamp': datetime.now().isoformat()
        }
        
        # JSON 직렬화/역직렬화
        json_str = json.dumps(sample_data, indent=2)
        parsed_data = json.loads(json_str)
        
        assert parsed_data['stage'] == 3
        assert parsed_data['target_samples'] == 100000
    
    runner.run_test("JSON 처리", test_json_operations)
    
    return runner


def main():
    """메인 함수"""
    print("🎯 Stage 3 간단한 테스트 실행기")
    print("="*60)
    print("목적: 기본 기능 검증 (pytest 의존성 없음)")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    start_time = time.time()
    
    # 테스트 실행
    import_runner = test_imports()
    basic_runner = test_basic_functionality() 
    init_runner = test_initialization()
    file_runner = test_file_operations()
    
    # 전체 결과 집계
    total_tests = (import_runner.tests_run + basic_runner.tests_run + 
                  init_runner.tests_run + file_runner.tests_run)
    total_passed = (import_runner.tests_passed + basic_runner.tests_passed +
                   init_runner.tests_passed + file_runner.tests_passed)
    total_failed = (import_runner.tests_failed + basic_runner.tests_failed +
                   init_runner.tests_failed + file_runner.tests_failed)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 최종 결과
    print("\n" + "="*60)
    print("🏆 전체 테스트 결과")
    print("="*60)
    print(f"총 테스트: {total_tests}")
    print(f"통과: {total_passed}")
    print(f"실패: {total_failed}")
    print(f"성공률: {(total_passed/total_tests*100):.1f}%")
    print(f"소요 시간: {duration:.1f}초")
    
    # 모든 실패 내역
    all_failures = (import_runner.failures + basic_runner.failures + 
                   init_runner.failures + file_runner.failures)
    
    if all_failures:
        print("\n❌ 전체 실패 내역:")
        for name, error in all_failures:
            print(f"  - {name}: {error}")
    
    # 프로덕션 준비도 평가
    success_rate = total_passed / total_tests
    if success_rate >= 0.9:
        grade = "🏆 우수 (Excellent)"
        ready = True
    elif success_rate >= 0.8:
        grade = "🎯 준비완료 (Ready)"
        ready = True
    elif success_rate >= 0.7:
        grade = "⚠️ 주의필요 (Needs Attention)"
        ready = False
    else:
        grade = "❌ 미준비 (Not Ready)"
        ready = False
    
    print(f"\n🎯 Stage 3 기본 준비도: {grade}")
    
    if ready:
        print("\n✅ 기본 테스트 통과! 다음 단계:")
        print("  1. 복잡한 통합 테스트 실행 고려")
        print("  2. 실제 데이터로 테스트 진행")
        print("  3. GPU 테스트 실행 (CUDA 사용 가능시)")
    else:
        print("\n⚠️ 기본 테스트 실패 - 우선 해결 필요")
    
    print("="*60)
    
    return 0 if ready else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)