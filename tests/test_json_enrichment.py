"""
JSON 파싱 및 EDI 추출 테스트

목적: preprocess의 JSON 파싱 및 클래스 맵 생성 검증
"""

import pandas as pd
import json
from pathlib import Path
import tempfile

from dataset.preprocess import preprocess
from pillsnap.stage1.utils import build_edi_classes, validate_class_map


class MockConfig:
    """테스트용 설정 객체"""

    def __init__(self):
        self.manifest_filename = "test_manifest.csv"
        self.quarantine_dirname = "quarantine"


def create_test_label_json(edi_code: str, mapping_code: str) -> dict:
    """테스트용 라벨 JSON 생성"""
    return {
        "images": [
            {
                "dl_mapping_code": mapping_code,
                "di_edi_code": edi_code,
                "drug_N": "1",
                "dl_name": "Test Drug",
                "drug_shape": "원형",
                "print_front": "TEST",
                "print_back": "123",
            }
        ]
    }


def test_preprocess_json_enrichment():
    """preprocess가 JSON을 파싱하여 EDI를 추출하는지 테스트"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 테스트 데이터 준비
        img_dir = tmpdir / "images"
        label_dir = tmpdir / "labels"
        img_dir.mkdir()
        label_dir.mkdir()

        # 이미지와 라벨 파일 생성
        test_data = []
        for i in range(3):
            # 이미지 파일 (빈 파일)
            img_path = img_dir / f"K-00{i:04d}_test.png"
            img_path.touch()

            # 라벨 JSON 파일
            label_path = label_dir / f"K-00{i:04d}_test.json"
            edi = f"65370006{i}"
            mapping = f"K-00{i:04d}"

            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(create_test_label_json(edi, mapping), f)

            test_data.append(
                {
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "code": f"{mapping}_test",
                    "is_pair": True,
                }
            )

        # DataFrame 생성
        df = pd.DataFrame(test_data)

        # preprocess 실행
        cfg = MockConfig()
        artifacts_dir = tmpdir / "artifacts"

        result_df = preprocess(df, cfg, artifacts_dir=str(artifacts_dir))

        # 검증: 새로운 컬럼들이 추가되었는지
        expected_cols = [
            "image_path",
            "label_path",
            "code",
            "is_pair",
            "mapping_code",
            "edi_code",
            "json_ok",
        ]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"

        # 검증: JSON 파싱이 성공했는지
        assert result_df["json_ok"].all(), "Some JSON parsing failed"

        # 검증: EDI 코드가 올바르게 추출되었는지
        assert len(result_df["edi_code"].dropna()) == 3, "EDI codes not extracted"
        assert result_df["edi_code"].iloc[0] == "653700060"
        assert result_df["edi_code"].iloc[1] == "653700061"
        assert result_df["edi_code"].iloc[2] == "653700062"

        # 검증: mapping_code가 올바르게 추출되었는지
        assert result_df["mapping_code"].iloc[0] == "K-000000"

        # 검증: CSV 파일이 생성되었는지
        manifest_path = artifacts_dir / cfg.manifest_filename
        assert manifest_path.exists(), "Manifest CSV not created"

        # CSV 다시 로드하여 스키마 확인
        loaded_df = pd.read_csv(manifest_path)
        for col in expected_cols:
            assert col in loaded_df.columns, f"Column {col} not preserved in CSV"


def test_build_edi_classes():
    """build_edi_classes 함수 테스트"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 테스트 매니페스트 생성
        test_data = pd.DataFrame(
            {
                "image_path": [f"/path/img{i}.png" for i in range(5)],
                "label_path": [f"/path/label{i}.json" for i in range(5)],
                "code": [f"K-00{i:04d}_test" for i in range(5)],
                "is_pair": [True] * 5,
                "edi_code": ["653700060", "653700061", "653700060", "653700062", None],
                "mapping_code": [f"K-00{i:04d}" for i in range(5)],
                "json_ok": [True, True, True, True, False],
            }
        )

        manifest_path = tmpdir / "test_manifest.csv"
        test_data.to_csv(manifest_path, index=False)

        # build_edi_classes 실행
        outfile = tmpdir / "classes.json"
        class_map = build_edi_classes(manifest_path, outfile)

        # 검증: 클래스 맵이 생성되었는지
        assert len(class_map) == 3, "Should have 3 unique EDI codes"

        # 검증: EDI가 정렬되어 있는지
        edi_list = list(class_map.keys())
        assert edi_list == sorted(edi_list), "EDI codes should be sorted"

        # 검증: class_id가 연속적인지
        assert list(class_map.values()) == [0, 1, 2], "Class IDs should be continuous"

        # 검증: 파일이 생성되었는지
        assert outfile.exists(), "Class map JSON not created"

        # JSON 다시 로드하여 확인
        with open(outfile, "r", encoding="utf-8") as f:
            loaded_map = json.load(f)
        assert loaded_map == class_map, "Saved and loaded class maps don't match"


def test_empty_dataframe_schema():
    """빈 DataFrame도 스키마를 유지하는지 테스트"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # 빈 DataFrame
        df = pd.DataFrame(columns=["image_path", "label_path", "code", "is_pair"])

        cfg = MockConfig()
        result_df = preprocess(df, cfg, artifacts_dir=tmpdir)

        # 검증: 모든 필수 컬럼이 있는지
        expected_cols = [
            "image_path",
            "label_path",
            "code",
            "is_pair",
            "mapping_code",
            "edi_code",
            "json_ok",
        ]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column in empty DF: {col}"

        # 검증: 행이 0개인지
        assert len(result_df) == 0, "Empty DF should have 0 rows"


def test_invalid_json_handling():
    """잘못된 JSON 처리 테스트"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 이미지와 잘못된 JSON 생성
        img_path = tmpdir / "test.png"
        label_path = tmpdir / "test.json"
        img_path.touch()

        # 잘못된 JSON
        with open(label_path, "w") as f:
            f.write("invalid json content")

        df = pd.DataFrame(
            [
                {
                    "image_path": str(img_path),
                    "label_path": str(label_path),
                    "code": "test_code",
                    "is_pair": True,
                }
            ]
        )

        cfg = MockConfig()
        result_df = preprocess(df, cfg, artifacts_dir=tmpdir)

        # 검증: json_ok가 False인지
        assert not result_df["json_ok"].iloc[0], "Invalid JSON should set json_ok=False"

        # 검증: edi_code가 None인지
        assert pd.isna(
            result_df["edi_code"].iloc[0]
        ), "Invalid JSON should have None edi_code"


def test_validate_class_map():
    """validate_class_map 함수 테스트"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 유효한 클래스 맵 생성
        valid_map = {"653700060": 0, "653700061": 1, "653700062": 2}
        valid_file = tmpdir / "valid.json"
        with open(valid_file, "w") as f:
            json.dump(valid_map, f)

        result = validate_class_map(valid_file)
        assert result["valid"], "Valid class map should pass validation"
        assert result["num_classes"] == 3

        # 잘못된 클래스 맵 (비연속 ID)
        invalid_map = {"653700060": 0, "653700061": 2, "653700062": 5}
        invalid_file = tmpdir / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump(invalid_map, f)

        result = validate_class_map(invalid_file)
        assert not result["valid"], "Invalid class map should fail validation"
        assert "not continuous" in str(result["issues"])
