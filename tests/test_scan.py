"""
dataset.scan 모듈 유닛 테스트

테스트 목적:
- scan_dataset 함수의 기본 동작 검증
- 이미지-라벨 매칭 정확도 확인
- 매칭/미매칭 카운트 정확성 검증
- DataFrame 컬럼 구조 및 데이터 타입 보장
- 통계 정보의 일관성 확인
- 중복 basename 경고 요약 출력 검증
"""

import pytest
from pathlib import Path
import sys
from io import StringIO
from contextlib import redirect_stdout

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.scan import scan_dataset, scan_dataset_summary, validate_scan_results


class TestScanDataset:
    """scan_dataset 함수 기본 동작 테스트"""
    
    def test_basic_scan_functionality(self, tmp_path):
        """기본 스캔 기능 테스트 - 매칭/미매칭 파일 정확한 처리"""
        # 테스트 데이터 생성
        tmp_root = tmp_path
        
        # 매칭 쌍 1개
        (tmp_root / "a.jpg").touch()
        (tmp_root / "a.json").touch()
        
        # 고아 이미지 1개 (라벨 없음)
        (tmp_root / "b.jpg").touch()
        
        # 고아 라벨 1개 (이미지 없음)  
        (tmp_root / "c.json").touch()
        
        # 스캔 실행
        df, stats = scan_dataset(tmp_root, [".jpg", ".jpeg", ".png"], ".json")
        
        # 기본 DataFrame 컬럼 구조 검증
        expected_columns = {"image_path", "label_path", "code", "is_pair"}
        assert expected_columns.issubset(set(df.columns)), f"Missing columns: {expected_columns - set(df.columns)}"
        
        # 통계 정확성 검증
        assert stats["pairs"] == 1, f"Expected 1 pair, got {stats['pairs']}"
        assert stats["images"] == 2, f"Expected 2 images, got {stats['images']}"
        assert stats["labels"] == 2, f"Expected 2 labels, got {stats['labels']}"
        assert stats["mismatches"] == 2, f"Expected 2 mismatches (b.jpg, c.json), got {stats['mismatches']}"
        
        # DataFrame 행 수 검증
        assert len(df) == 3, f"Expected 3 rows (a, b, c), got {len(df)}"
        
        # 매칭 쌍 검증
        matched_rows = df[df["is_pair"] == True]
        assert len(matched_rows) == 1, f"Expected 1 matched pair, got {len(matched_rows)}"
        
        # 고아 파일 검증
        orphaned_rows = df[df["is_pair"] == False]
        assert len(orphaned_rows) == 2, f"Expected 2 orphaned files, got {len(orphaned_rows)}"
    
    def test_empty_directory_scan(self, tmp_path):
        """빈 디렉토리 스캔 테스트"""
        df, stats = scan_dataset(tmp_path, [".jpg", ".png"], ".json")
        
        # 빈 결과 검증
        assert len(df) == 0, "Empty directory should return empty DataFrame"
        assert stats["images"] == 0, "Empty directory should have 0 images"
        assert stats["labels"] == 0, "Empty directory should have 0 labels"
        assert stats["pairs"] == 0, "Empty directory should have 0 pairs"
        assert stats["mismatches"] == 0, "Empty directory should have 0 mismatches"
    
    def test_only_images_scan(self, tmp_path):
        """이미지만 있는 디렉토리 스캔"""
        # 이미지 파일들만 생성
        (tmp_path / "img1.jpg").touch()
        (tmp_path / "img2.png").touch()
        (tmp_path / "img3.jpeg").touch()
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".png", ".jpeg"], ".json")
        
        # 결과 검증
        assert stats["images"] == 3, "Should detect 3 images"
        assert stats["labels"] == 0, "Should detect 0 labels"
        assert stats["pairs"] == 0, "Should have 0 pairs"
        assert stats["mismatches"] == 3, "Should have 3 mismatches (all orphaned images)"
        
        # 모든 파일이 고아 이미지여야 함
        assert (df["is_pair"] == False).all(), "All files should be orphaned"
        assert df["image_path"].notna().all(), "All rows should have image_path"
        assert df["label_path"].isna().all(), "All rows should have null label_path"
    
    def test_only_labels_scan(self, tmp_path):
        """라벨만 있는 디렉토리 스캔"""
        # 라벨 파일들만 생성
        (tmp_path / "label1.json").touch()
        (tmp_path / "label2.json").touch()
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".png"], ".json")
        
        # 결과 검증
        assert stats["images"] == 0, "Should detect 0 images"
        assert stats["labels"] == 2, "Should detect 2 labels"
        assert stats["pairs"] == 0, "Should have 0 pairs"
        assert stats["mismatches"] == 2, "Should have 2 mismatches (all orphaned labels)"
        
        # 모든 파일이 고아 라벨이어야 함
        assert (df["is_pair"] == False).all(), "All files should be orphaned"
        assert df["image_path"].isna().all(), "All rows should have null image_path"
        assert df["label_path"].notna().all(), "All rows should have label_path"
    
    def test_subdirectory_scan(self, tmp_path):
        """서브디렉토리 포함 스캔 테스트"""
        # 루트 레벨 파일
        (tmp_path / "root.jpg").touch()
        (tmp_path / "root.json").touch()
        
        # 서브디렉토리 생성 및 파일 추가
        sub1 = tmp_path / "subdir1"
        sub1.mkdir()
        (sub1 / "sub1.png").touch()
        (sub1 / "sub1.json").touch()
        
        sub2 = tmp_path / "subdir2" 
        sub2.mkdir()
        (sub2 / "sub2.jpeg").touch()
        # sub2에는 라벨 없음 (고아 이미지)
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".png", ".jpeg"], ".json")
        
        # 통계 검증
        assert stats["images"] == 3, "Should detect 3 images across all subdirectories"
        assert stats["labels"] == 2, "Should detect 2 labels"
        assert stats["pairs"] == 2, "Should have 2 matched pairs"
        assert stats["mismatches"] == 1, "Should have 1 mismatch (sub2.jpeg)"
        
        # DataFrame 구조 검증
        assert len(df) == 3, "Should have 3 total file entries"
        
        # 경로가 올바르게 저장되는지 확인
        image_paths = df[df["image_path"].notna()]["image_path"].tolist()
        assert any("root.jpg" in path for path in image_paths), "Should contain root.jpg path"
        assert any("sub1.png" in path for path in image_paths), "Should contain sub1.png path"
        assert any("sub2.jpeg" in path for path in image_paths), "Should contain sub2.jpeg path"


class TestScanValidation:
    """scan 결과 검증 테스트"""
    
    def test_dataframe_column_types(self, tmp_path):
        """DataFrame 컬럼 타입 검증"""
        # 테스트 데이터 생성
        (tmp_path / "test.jpg").touch()
        (tmp_path / "test.json").touch()
        
        df, stats = scan_dataset(tmp_path, [".jpg"], ".json")
        
        # 컬럼 존재 확인
        required_columns = ["image_path", "label_path", "code", "is_pair"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"
        
        # 데이터 타입 확인
        assert df["code"].dtype == object, "code column should be string type"
        assert df["is_pair"].dtype == bool, "is_pair column should be boolean type"
        
        # 값 형식 확인
        for idx, row in df.iterrows():
            if row["image_path"] is not None:
                assert isinstance(row["image_path"], str), "image_path should be string when not null"
                assert Path(row["image_path"]).exists(), "image_path should point to existing file"
            
            if row["label_path"] is not None:
                assert isinstance(row["label_path"], str), "label_path should be string when not null"
                assert Path(row["label_path"]).exists(), "label_path should point to existing file"
            
            assert isinstance(row["code"], str), "code should always be string"
            assert isinstance(row["is_pair"], bool), "is_pair should always be boolean"
    
    def test_statistics_consistency(self, tmp_path):
        """통계 정보 일관성 검증"""
        # 복잡한 테스트 케이스 생성
        for i in range(3):
            (tmp_path / f"pair_{i}.jpg").touch()
            (tmp_path / f"pair_{i}.json").touch()
        
        for i in range(2):
            (tmp_path / f"orphan_img_{i}.png").touch()
        
        for i in range(1):
            (tmp_path / f"orphan_label_{i}.json").touch()
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".png"], ".json")
        
        # validate_scan_results 함수로 일관성 검증
        issues = validate_scan_results(df, stats)
        assert len(issues) == 0, f"Validation issues found: {issues}"
        
        # 수동 일관성 검증
        assert stats["pairs"] + stats["orphaned_images"] == stats["images"], "Image count mismatch"
        assert stats["pairs"] + stats["orphaned_labels"] == stats["labels"], "Label count mismatch"
        assert stats["orphaned_images"] + stats["orphaned_labels"] == stats["mismatches"], "Mismatch count error"
        
        # DataFrame과 통계 일치 검증
        df_pairs = (df["is_pair"] == True).sum()
        df_mismatches = (df["is_pair"] == False).sum()
        assert df_pairs == stats["pairs"], "DataFrame pairs don't match stats"
        assert df_mismatches == stats["mismatches"], "DataFrame mismatches don't match stats"
    
    def test_file_extension_handling(self, tmp_path):
        """파일 확장자 대소문자 처리 테스트"""
        # 다양한 대소문자 조합
        (tmp_path / "test.JPG").touch()      # 대문자
        (tmp_path / "test.Json").touch()     # 혼합
        (tmp_path / "other.jpg").touch()     # 소문자
        (tmp_path / "other.JSON").touch()    # 대문자
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".JPG"], ".json")
        
        # 대소문자 관계없이 정확히 매칭되어야 함
        assert stats["pairs"] == 2, "Should match regardless of case"
        assert stats["images"] == 2, "Should detect both JPG and jpg"
        assert stats["labels"] == 2, "Should detect both Json and JSON"
        assert stats["mismatches"] == 0, "All files should be matched"


class TestScanEdgeCases:
    """예외 상황 및 엣지 케이스 테스트"""
    
    def test_duplicate_basenames(self, tmp_path):
        """동일한 basename을 가진 파일들 처리"""
        # 같은 basename의 서로 다른 확장자
        (tmp_path / "duplicate.jpg").touch()
        (tmp_path / "duplicate.png").touch()  # 같은 basename
        (tmp_path / "duplicate.json").touch()
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".png"], ".json")
        
        # 첫 번째 발견된 이미지만 사용되어야 함 (로그에 경고 출력)
        assert stats["images"] == 1, "Should count only first image (duplicate ignored)"
        assert stats["labels"] == 1, "Should count the label"
        assert len(df) == 1, "Should have only one entry (duplicate basename)"
        assert stats["pairs"] == 1, "Should have one matched pair"
        assert stats["mismatches"] == 0, "Should have no mismatches"
        
    def test_duplicate_basenames_warning_summary(self, tmp_path):
        """중복 basename 경고 요약 출력 검증"""
        # 중복 basename 파일들 생성
        (tmp_path / "dup1.jpg").touch()
        (tmp_path / "dup1.png").touch()  # 같은 basename
        (tmp_path / "dup1.json").touch()
        
        (tmp_path / "dup2.jpg").touch() 
        (tmp_path / "dup2.jpeg").touch()  # 같은 basename
        (tmp_path / "dup2.json").touch()
        
        # 라벨 중복 생성
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "dup2.json").touch()  # 같은 basename으로 라벨 중복
        
        # stdout 캡처하여 경고 요약 확인
        stdout_capture = StringIO()
        with redirect_stdout(stdout_capture):
            df, stats = scan_dataset(tmp_path, [".jpg", ".png", ".jpeg"], ".json")
        
        output = stdout_capture.getvalue()
        
        # 요약 출력 형식 검증
        assert "=== Duplicate Basename Summary ===" in output, "Missing duplicate summary header"
        assert "Duplicate image basenames" in output, "Missing image duplicates section"
        assert "Duplicate label basenames" in output, "Missing label duplicates section"
        assert "dup1: 1 duplicates" in output, "Missing dup1 count"
        assert "dup2: 1 duplicates" in output, "Missing dup2 count"
        assert "Total unique duplicate image basenames:" in output, "Missing total image summary"
        assert "Total unique duplicate label basenames:" in output, "Missing total label summary"
        assert "=================================" in output, "Missing summary footer"
        
        # 실제 데이터 검증
        assert len(df) == 2, "Should have 2 unique basenames despite duplicates"
    
    def test_nonexistent_directory(self):
        """존재하지 않는 디렉토리 처리"""
        with pytest.raises(FileNotFoundError):
            scan_dataset("/path/that/does/not/exist", [".jpg"], ".json")
    
    def test_special_characters_in_filenames(self, tmp_path):
        """파일명에 특수문자가 있는 경우"""
        # 특수문자가 포함된 파일명
        (tmp_path / "file with spaces.jpg").touch()
        (tmp_path / "file with spaces.json").touch()
        (tmp_path / "file-with-dashes.png").touch()
        (tmp_path / "file-with-dashes.json").touch()  # 추가
        (tmp_path / "file_with_underscores.jpg").touch()  # 확장자 변경
        (tmp_path / "file_with_underscores.json").touch()
        (tmp_path / "file.with.dots.jpeg").touch()
        (tmp_path / "file.with.dots.json").touch()
        
        df, stats = scan_dataset(tmp_path, [".jpg", ".png", ".jpeg"], ".json")
        
        # 특수문자가 있어도 정상적으로 매칭되어야 함
        assert stats["pairs"] == 4, f"Should match files with special characters. Got pairs={stats['pairs']}, images={stats['images']}, labels={stats['labels']}"
        assert stats["mismatches"] == 0, f"All files should be matched. Got {stats['mismatches']}"
        
        # code에 특수문자가 올바르게 보존되어야 함
        codes = set(df["code"].tolist())
        expected_codes = {"file with spaces", "file-with-dashes", "file_with_underscores", "file.with.dots"}
        assert expected_codes.issubset(codes), "Special characters in codes should be preserved"


class TestScanSummary:
    """scan_dataset_summary 함수 테스트"""
    
    def test_summary_generation(self, tmp_path):
        """요약 리포트 생성 테스트"""
        # 테스트 데이터 생성
        (tmp_path / "matched.jpg").touch()
        (tmp_path / "matched.json").touch()
        (tmp_path / "orphan.png").touch()
        
        summary = scan_dataset_summary(tmp_path, [".jpg", ".png"], ".json", max_examples=5)
        
        # 요약에 포함되어야 할 정보들
        assert "Dataset Scan Summary" in summary, "Should contain title"
        assert str(tmp_path) in summary, "Should contain root directory path"
        assert "Images: 2" in summary, "Should show correct image count"
        assert "Labels: 1" in summary, "Should show correct label count"
        assert "Matched Pairs: 1" in summary, "Should show correct pair count"
        assert "Orphaned Files: 1" in summary, "Should show correct orphan count"
        assert "matched" in summary, "Should show matched file example"
        assert "orphan" in summary, "Should show orphaned file example"
    
    def test_summary_with_no_files(self, tmp_path):
        """파일이 없는 경우 요약 테스트"""
        summary = scan_dataset_summary(tmp_path, [".jpg"], ".json")
        
        assert "Images: 0" in summary, "Should show 0 images"
        assert "Labels: 0" in summary, "Should show 0 labels"
        assert "Matched Pairs: 0" in summary, "Should show 0 pairs"


class TestIntegration:
    """통합 테스트"""
    
    def test_full_workflow(self, tmp_path):
        """전체 워크플로우 통합 테스트"""
        # 복잡한 디렉토리 구조 생성
        sub1 = tmp_path / "dataset1"
        sub2 = tmp_path / "dataset2"
        sub1.mkdir()
        sub2.mkdir()
        
        # 다양한 케이스의 파일들
        test_cases = [
            # 완전한 쌍들
            ("sample_001.jpg", "sample_001.json", sub1),
            ("sample_002.png", "sample_002.json", sub1),
            ("sample_003.jpeg", "sample_003.json", sub2),
            
            # 고아 이미지들
            ("orphan_img_001.jpg", None, sub1),
            ("orphan_img_002.png", None, sub2),
            
            # 고아 라벨들
            (None, "orphan_label_001.json", sub1),
            (None, "orphan_label_002.json", sub2),
        ]
        
        for img_file, label_file, directory in test_cases:
            if img_file:
                (directory / img_file).touch()
            if label_file:
                (directory / label_file).touch()
        
        # 스캔 실행
        df, stats = scan_dataset(tmp_path, [".jpg", ".png", ".jpeg"], ".json")
        
        # 전체 결과 검증
        expected_stats = {
            "images": 5,        # 3 pairs + 2 orphan images
            "labels": 5,        # 3 pairs + 2 orphan labels
            "pairs": 3,         # 3 matched pairs
            "mismatches": 4,    # 2 orphan images + 2 orphan labels
            "orphaned_images": 2,
            "orphaned_labels": 2
        }
        
        for key, expected_value in expected_stats.items():
            assert stats[key] == expected_value, f"Stats mismatch for {key}: expected {expected_value}, got {stats[key]}"
        
        # 검증 실행
        issues = validate_scan_results(df, stats)
        assert len(issues) == 0, f"Validation failed: {issues}"
        
        # 요약 생성 테스트
        summary = scan_dataset_summary(tmp_path, [".jpg", ".png", ".jpeg"], ".json")
        assert "Dataset Scan Summary" in summary, "Summary should be generated successfully"