"""
Stage 3 Manifest Creator 단위 테스트

테스트 범위:
- Stage3ManifestCreator 초기화 및 설정 검증
- 데이터 스캔 기능 테스트
- 클래스 선택 로직 검증
- 샘플링 균형성 테스트
- Train/Val 분할 정확성
- 파일 저장 및 무결성 검증
- Edge case 처리
"""

import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.create_stage3_manifest import Stage3ManifestCreator


class TestStage3ManifestCreator:
    """Stage3ManifestCreator 테스트"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """임시 데이터 디렉토리 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            
            # 디렉토리 구조 생성
            single_dir = data_root / "train/images/single/TS_1_single"
            combo_dir = data_root / "train/images/combination/TS_1_combo"
            single_dir.mkdir(parents=True)
            combo_dir.mkdir(parents=True)
            
            # 더미 이미지 파일 생성 (1500개 클래스, 각 100-150개 이미지)
            for i in range(1500):
                k_code = f"K{i:06d}"
                
                # Single 이미지
                k_single_dir = single_dir.parent / f"TS_{i%10+1}_single" / k_code
                k_single_dir.mkdir(parents=True, exist_ok=True)
                
                n_single = random.randint(60, 100)
                for j in range(n_single):
                    img_file = k_single_dir / f"img_{j:04d}.jpg"
                    img_file.touch()
                
                # Combination 이미지
                k_combo_dir = combo_dir.parent / f"TS_{i%8+1}_combo" / k_code
                k_combo_dir.mkdir(parents=True, exist_ok=True)
                
                n_combo = random.randint(40, 80)
                for j in range(n_combo):
                    img_file = k_combo_dir / f"img_{j:04d}.jpg"
                    img_file.touch()
            
            yield data_root
    
    @pytest.fixture
    def creator(self, temp_data_dir):
        """Stage3ManifestCreator 인스턴스"""
        return Stage3ManifestCreator(data_root=str(temp_data_dir))
    
    def test_initialization(self, creator):
        """초기화 및 설정 검증 (Classification 중심)"""
        assert creator.target_samples == 100000
        assert creator.target_classes == 1000
        assert creator.samples_per_class == 100
        assert creator.single_ratio == 0.95  # Classification 중심 비율
        assert creator.train_ratio == 0.8
        assert creator.val_ratio == 0.2
        assert creator.seed == 42
    
    def test_scan_available_data(self, creator):
        """데이터 스캔 기능 테스트"""
        available_classes = creator.scan_available_data()
        
        # 충분한 클래스가 스캔되었는지 확인
        assert len(available_classes) >= 1000
        
        # 각 클래스에 single과 combination 데이터가 있는지 확인
        for k_code, data in list(available_classes.items())[:10]:
            assert 'single' in data
            assert 'combination' in data
            assert len(data['single']) > 0
            assert len(data['combination']) > 0
    
    def test_select_target_classes(self, creator):
        """클래스 선택 로직 검증"""
        creator.scan_available_data()
        selected_classes = creator.select_target_classes()
        
        # 정확히 1000개 클래스가 선택되었는지 확인
        assert len(selected_classes) == 1000
        
        # 중복이 없는지 확인
        assert len(set(selected_classes)) == len(selected_classes)
        
        # 선택된 클래스가 모두 사용 가능한 클래스인지 확인
        for k_code in selected_classes:
            assert k_code in creator.available_classes
    
    def test_sample_images_for_class(self, creator):
        """클래스별 샘플링 검증"""
        creator.scan_available_data()
        creator.select_target_classes()
        
        # 첫 번째 클래스에서 샘플링
        k_code = creator.selected_classes[0]
        records = creator.sample_images_for_class(k_code)
        
        # 정확히 100개 샘플이 선택되었는지 확인
        assert len(records) == creator.samples_per_class
        
        # Single/Combination 비율 확인 (약 60:40, ±10% 허용)
        single_count = sum(1 for r in records if r['image_type'] == 'single')
        combo_count = sum(1 for r in records if r['image_type'] == 'combination')
        
        expected_single = creator.samples_per_class * creator.single_ratio
        assert abs(single_count - expected_single) <= 20  # ±20% 허용
        
        # 모든 레코드가 올바른 필드를 가지는지 확인
        for record in records:
            assert 'image_path' in record
            assert 'mapping_code' in record
            assert 'image_type' in record
            assert 'source' in record
            assert record['mapping_code'] == k_code
            assert record['image_type'] in ['single', 'combination']
            assert record['source'] == 'train'
            assert Path(record['image_path']).exists()
    
    def test_create_manifest(self, creator):
        """전체 Manifest 생성 테스트"""
        df = creator.create_manifest()
        
        # DataFrame 크기 확인
        assert len(df) > 0
        assert len(df) <= creator.target_samples
        
        # 필수 컬럼 확인
        required_columns = ['image_path', 'mapping_code', 'image_type', 'source']
        for col in required_columns:
            assert col in df.columns
        
        # 클래스 수 확인
        unique_classes = df['mapping_code'].nunique()
        assert unique_classes <= creator.target_classes
        
        # Single/Combination 비율 확인 (Classification 중심: 95:5)
        single_ratio = (df['image_type'] == 'single').mean()
        assert 0.90 <= single_ratio <= 1.00  # 95% ± 5%
    
    def test_split_train_val(self, creator):
        """Train/Val 분할 테스트"""
        df = creator.create_manifest()
        train_df, val_df = creator.split_train_val(df)
        
        # 분할 비율 확인
        total = len(train_df) + len(val_df)
        assert abs(len(train_df) / total - creator.train_ratio) < 0.05
        assert abs(len(val_df) / total - creator.val_ratio) < 0.05
        
        # 데이터 누락 없음 확인
        assert total == len(df)
        
        # 클래스 균형 확인 (각 클래스가 train/val에 모두 존재)
        train_classes = set(train_df['mapping_code'].unique())
        val_classes = set(val_df['mapping_code'].unique())
        
        # 대부분의 클래스가 양쪽에 존재해야 함
        overlap = train_classes & val_classes
        assert len(overlap) >= len(train_classes) * 0.95
    
    def test_save_manifests(self, creator):
        """파일 저장 테스트"""
        df = creator.create_manifest()
        train_df, val_df = creator.split_train_val(df)
        
        # 파일 저장
        train_path, val_path, mapping_path, stats_path = creator.save_manifests(train_df, val_df)
        
        # 파일 존재 확인
        assert train_path.exists()
        assert val_path.exists()
        assert mapping_path.exists()
        assert stats_path.exists()
        
        # CSV 파일 로드 및 검증
        loaded_train = pd.read_csv(train_path)
        loaded_val = pd.read_csv(val_path)
        
        assert len(loaded_train) == len(train_df)
        assert len(loaded_val) == len(val_df)
        
        # JSON 파일 로드 및 검증
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
        assert isinstance(class_mapping, dict)
        assert len(class_mapping) > 0
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        assert 'timestamp' in stats
        assert stats['stage'] == 3
        assert stats['total_samples'] == len(train_df) + len(val_df)
    
    def test_edge_case_insufficient_classes(self):
        """불충분한 클래스 수 처리 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            
            # 적은 수의 클래스만 생성 (500개)
            single_dir = data_root / "train/images/single/TS_1_single"
            single_dir.mkdir(parents=True)
            
            for i in range(500):
                k_code = f"K{i:06d}"
                k_dir = single_dir / k_code
                k_dir.mkdir()
                
                # 충분한 이미지 생성
                for j in range(100):
                    img_file = k_dir / f"img_{j:04d}.jpg"
                    img_file.touch()
            
            creator = Stage3ManifestCreator(data_root=str(data_root))
            creator.scan_available_data()
            selected_classes = creator.select_target_classes()
            
            # 사용 가능한 만큼만 선택되었는지 확인
            assert len(selected_classes) == 500
            assert creator.target_classes == 500  # 자동 조정됨
    
    def test_reproducibility(self, creator):
        """재현성 테스트 (동일 시드 → 동일 결과)"""
        # 첫 번째 실행
        df1 = creator.create_manifest()
        train_df1, val_df1 = creator.split_train_val(df1)
        
        # 두 번째 실행 (새 인스턴스)
        creator2 = Stage3ManifestCreator(data_root=str(creator.data_root))
        df2 = creator2.create_manifest()
        train_df2, val_df2 = creator2.split_train_val(df2)
        
        # 동일한 결과인지 확인
        assert len(df1) == len(df2)
        assert len(train_df1) == len(train_df2)
        assert len(val_df1) == len(val_df2)
        
        # 선택된 클래스가 동일한지 확인
        assert set(df1['mapping_code'].unique()) == set(df2['mapping_code'].unique())
    
    def test_performance_with_large_dataset(self, creator):
        """대용량 데이터셋 성능 테스트"""
        import time
        
        start_time = time.time()
        
        # 전체 프로세스 실행
        creator.scan_available_data()
        creator.select_target_classes()
        df = creator.create_manifest()
        train_df, val_df = creator.split_train_val(df)
        
        elapsed_time = time.time() - start_time
        
        # 프로덕션 환경에서 3분 이내에 완료되어야 함 (강화된 요구사항)
        assert elapsed_time < 180, f"처리 시간 초과: {elapsed_time:.1f}초 > 180초"
        
        print(f"✅ 성능 테스트: {len(df):,}개 샘플 처리 시간 = {elapsed_time:.2f}초")
    
    def test_run_method_integration(self):
        """run() 메서드 통합 테스트"""
        import tempfile
        from src.data.create_stage3_manifest import Stage3ManifestCreator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock 데이터 생성
            self._create_mock_data_structure_simple(tmpdir)
            creator = Stage3ManifestCreator(data_root=tmpdir)
            # 테스트를 위해 타겟 값을 작게 조정
            creator.target_samples = 100
            creator.target_classes = 10
            creator.samples_per_class = 10
            
            results = creator.run()
        
        # 모든 결과 파일이 반환되었는지 확인
        assert 'train_manifest' in results
        assert 'val_manifest' in results  
        assert 'class_mapping' in results
        assert 'sampling_report' in results
        
        # 모든 파일이 존재하는지 확인
        for name, path in results.items():
            assert path.exists(), f"{name} 파일이 존재하지 않음: {path}"
        
        # Train manifest 검증
        train_df = pd.read_csv(results['train_manifest'])
        assert len(train_df) > 0
        assert 'image_path' in train_df.columns
        assert 'mapping_code' in train_df.columns
        
        # 통계 리포트 검증
        with open(results['sampling_report'], 'r') as f:
            stats = json.load(f)
        assert stats['stage'] == 3
        assert stats['num_classes'] <= 1000
        
        # 프로덕션 메트릭 검증
        self._verify_production_metrics(stats)
    
    def _create_mock_data_structure_simple(self, base_dir):
        """간단한 Mock 데이터 구조 생성 (run 테스트용)"""
        from pathlib import Path
        
        base_path = Path(base_dir)
        
        # 10개 클래스 Mock 데이터 생성
        for i in range(10):
            k_code = f"K{i:06d}"
            
            # Single 이미지
            single_dir = base_path / "train" / "images" / "single" / f"TS_{i}_single" / k_code
            single_dir.mkdir(parents=True, exist_ok=True)
            for j in range(85):  # 클래스당 85개 (충분함)
                img_file = single_dir / f"{k_code}_{j:03d}.jpg"
                img_file.write_text(f"mock image {i}_{j}")
            
            # Combination 이미지 (적게, 하지만 존재)
            combo_dir = base_path / "train" / "images" / "combination" / f"TS_{i}_combo" / k_code
            combo_dir.mkdir(parents=True, exist_ok=True)
            for j in range(5):  # 클래스당 5개
                img_file = combo_dir / f"{k_code}_combo_{j:03d}.jpg"
                img_file.write_text(f"mock combo {i}_{j}")
    
    def _verify_production_metrics(self, stats):
        """프로덕션 메트릭 검증"""
        # 기본 구조 확인
        assert 'stage' in stats
        assert 'num_classes' in stats
        assert 'total_samples' in stats
        assert 'single_ratio' in stats
        assert 'combination_ratio' in stats
        
        # 데이터 품질 확인
        assert stats['num_classes'] > 0
        assert stats['total_samples'] > 0
        
        # Classification 중심 비율 검증
        single_ratio = stats['single_ratio']
        combo_ratio = stats['combination_ratio']
        
        assert single_ratio >= 0.85, f"Single 비율 부족: {single_ratio:.3f} < 0.85"
        assert combo_ratio <= 0.15, f"Combination 비율 초과: {combo_ratio:.3f} > 0.15"
        
        # 비율 합계 검증 (약간의 오차 허용)
        total_ratio = single_ratio + combo_ratio
        assert abs(total_ratio - 1.0) < 0.01, f"비율 합계 오류: {total_ratio:.3f} != 1.0"
        
        print(f"✅ Production metrics: Single {single_ratio:.1%}, Combination {combo_ratio:.1%}")


@pytest.mark.integration
class TestStage3ManifestCreatorIntegration:
    """Stage 3 Manifest Creator 통합 테스트"""
    
    @pytest.fixture
    def creator(self):
        """테스트용 Stage3ManifestCreator 인스턴스"""
        import tempfile
        from src.data.create_stage3_manifest import Stage3ManifestCreator
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock 데이터 생성
            self._create_mock_data_structure(tmpdir)
            creator = Stage3ManifestCreator(data_root=tmpdir)
            yield creator
    
    def _create_mock_data_structure(self, base_dir):
        """Mock 데이터 구조 생성"""
        from pathlib import Path
        
        base_path = Path(base_dir)
        
        # 10개 클래스 Mock 데이터 생성
        for i in range(10):
            k_code = f"K{i:06d}"
            
            # Single 이미지
            single_dir = base_path / "train" / "images" / "single" / f"TS_{i}_single" / k_code
            single_dir.mkdir(parents=True, exist_ok=True)
            for j in range(120):  # 클래스당 120개
                img_file = single_dir / f"{k_code}_{j:03d}.jpg"
                img_file.write_text(f"mock image {i}_{j}")
            
            # Combination 이미지 (적게)
            combo_dir = base_path / "train" / "images" / "combination" / f"TS_{i}_combo" / k_code
            combo_dir.mkdir(parents=True, exist_ok=True)
            for j in range(8):  # 클래스당 8개
                img_file = combo_dir / f"{k_code}_combo_{j:03d}.jpg"
                img_file.write_text(f"mock combo {i}_{j}")
    
    def test_dry_run_mode(self, capsys):
        """Dry run 모드 테스트"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 충분한 데이터 생성 (최소 80개 이미지)
            data_root = Path(tmpdir)
            single_dir = data_root / "train/images/single/TS_1_single"
            single_dir.mkdir(parents=True)
            
            for i in range(10):
                k_dir = single_dir / f"K{i:06d}"
                k_dir.mkdir()
                # 충분한 이미지 생성 (최소 요구사항 충족)
                for j in range(85):
                    (k_dir / f"img_{j:04d}.jpg").touch()
            
            # Dry run 실행
            creator = Stage3ManifestCreator(data_root=str(data_root))
            available = creator.scan_available_data()
            selected = creator.select_target_classes()
            
            # 출력 확인
            assert len(available) == 10
            assert len(selected) == 10  # 사용 가능한 만큼만
    
    def test_classification_focused_ratio(self, creator):
        """Classification 중심 비율 검증"""
        # Manifest 생성
        df = creator.create_manifest()
        
        # Single/Combination 비율 계산
        single_count = (df['image_type'] == 'single').sum()
        combo_count = (df['image_type'] == 'combination').sum()
        total_count = len(df)
        
        single_ratio = single_count / total_count
        combo_ratio = combo_count / total_count
        
        # Classification 중심 비율 검증 (95:5 ± 5%)
        assert single_ratio >= 0.90, f"Single 비율 부족: {single_ratio:.3f} < 0.90"
        assert single_ratio <= 1.00, f"Single 비율 초과: {single_ratio:.3f} > 1.00"
        assert combo_ratio >= 0.00, f"Combination 비율 음수: {combo_ratio:.3f}"
        assert combo_ratio <= 0.10, f"Combination 비율 초과: {combo_ratio:.3f} > 0.10"
        
        print(f"✅ 비율 검증: Single {single_ratio:.1%}, Combination {combo_ratio:.1%}")
        
        # 프로덕션 안정성 추가 검증
        self._verify_production_readiness(df)
    
    def test_realistic_combination_limit(self, creator):
        """현실적 Combination 데이터 제한 테스트"""
        # 각 클래스별 샘플링 테스트
        creator.scan_available_data()
        creator.select_target_classes()
        
        # 첫 번째 클래스로 테스트
        k_code = creator.selected_classes[0]
        records = creator.sample_images_for_class(k_code)
        
        # Combination 이미지가 5개를 초과하지 않는지 확인
        combo_records = [r for r in records if r['image_type'] == 'combination']
        assert len(combo_records) <= 5, f"Combination 이미지 과다: {len(combo_records)} > 5"
        
        # Single 이미지로 대부분 채워졌는지 확인
        single_records = [r for r in records if r['image_type'] == 'single']
        assert len(single_records) >= 95, f"Single 이미지 부족: {len(single_records)} < 95"
        
        print(f"✅ 현실적 제한: Single {len(single_records)}, Combination {len(combo_records)}")
        
        # 프로덕션 데이터 품질 검증
        self._verify_data_quality(records)


    def _create_mock_data_structure_simple(self, base_dir):
        """간단한 Mock 데이터 구조 생성 (run 테스트용)"""
        from pathlib import Path
        
        base_path = Path(base_dir)
        
        # 10개 클래스 Mock 데이터 생성
        for i in range(10):
            k_code = f"K{i:06d}"
            
            # Single 이미지
            single_dir = base_path / "train" / "images" / "single" / f"TS_{i}_single" / k_code
            single_dir.mkdir(parents=True, exist_ok=True)
            for j in range(15):  # 클래스당 15개
                img_file = single_dir / f"{k_code}_{j:03d}.jpg"
                img_file.write_text(f"mock image {i}_{j}")
            
            # Combination 이미지 (적게)
            combo_dir = base_path / "train" / "images" / "combination" / f"TS_{i}_combo" / k_code
            combo_dir.mkdir(parents=True, exist_ok=True)
            for j in range(3):  # 클래스당 3개
                img_file = combo_dir / f"{k_code}_combo_{j:03d}.jpg"
                img_file.write_text(f"mock combo {i}_{j}")
    
    def _verify_production_readiness(self, df: pd.DataFrame):
        """프로덕션 준비성 검증"""
        # 클래스 불균형 검증
        class_counts = df['mapping_code'].value_counts()
        min_samples = class_counts.min()
        max_samples = class_counts.max()
        imbalance_ratio = max_samples / min_samples
        
        assert imbalance_ratio <= 2.0, f"클래스 불균형 과도: {imbalance_ratio:.1f} > 2.0"
        
        # 데이터 분포 안정성
        single_per_class = df[df['image_type'] == 'single'].groupby('mapping_code').size()
        combo_per_class = df[df['image_type'] == 'combination'].groupby('mapping_code').size().reindex(single_per_class.index, fill_value=0)
        
        # 모든 클래스가 최소 95개 이상의 Single 이미지를 가져야 함
        assert single_per_class.min() >= 95, f"Single 이미지 부족 클래스 존재: 최소 {single_per_class.min()}개"
        
        # 파일 경로 유효성 (샘플링)
        sample_paths = df.sample(min(100, len(df)))['image_path']
        for path in sample_paths:
            assert Path(path).exists(), f"파일 경로 무효: {path}"
    
    def _verify_data_quality(self, records: list):
        """데이터 품질 검증"""
        # 중복 제거 확인
        paths = [r['image_path'] for r in records]
        assert len(paths) == len(set(paths)), "중복 이미지 경로 발견"
        
        # 파일 크기 검증 (비어있지 않은 파일)
        for record in records:
            path = Path(record['image_path'])
            assert path.stat().st_size > 0, f"빈 파일: {path}"
    
    def _verify_production_metrics(self, stats: dict):
        """프로덕션 메트릭 검증"""
        # 필수 메트릭 존재 확인
        required_keys = ['timestamp', 'stage', 'total_samples', 'num_classes', 'single_ratio']
        for key in required_keys:
            assert key in stats, f"필수 메트릭 누락: {key}"
        
        # 메트릭 범위 검증
        assert 0.9 <= stats['single_ratio'] <= 1.0, f"Single 비율 범위 초과: {stats['single_ratio']}"
        assert stats['total_samples'] >= 10000, f"샘플 수 부족: {stats['total_samples']}"
        assert 100 <= stats['num_classes'] <= 1000, f"클래스 수 범위 초과: {stats['num_classes']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])