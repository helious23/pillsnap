"""
목적:
  - Stage 1 확장 매니페스트(artifacts/manifest_enriched.csv)를 로드해
    이미지 경로와 edi_code를 사용한 분류용 Dataset 제공
입력:
  - manifest_csv: artifacts/manifest_enriched.csv
  - classes_json: artifacts/classes_step11.json (edi_code -> class_id)
출력:
  - __getitem__: (image_tensor, class_id)
검증 포인트:
  - CSV 스키마: ['image_path','label_path','code','is_pair','mapping_code','edi_code','json_ok', ...]
  - 파일 존재성 확인 옵션
"""

from pathlib import Path
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PillsnapClsDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str,
        classes_json: str,
        transform=None,
        require_exists: bool = True,
    ):
        """
        EDI 코드 기반 분류용 데이터셋

        Args:
            manifest_csv: Stage 1에서 생성된 enriched manifest 파일
            classes_json: EDI 코드 -> class_id 매핑 파일
            transform: 이미지 전처리 변환
            require_exists: 파일 존재성 체크 여부
        """
        self.manifest_csv = Path(manifest_csv)
        self.classes_json = Path(classes_json)
        self.transform = transform
        self.require_exists = require_exists

        # 파일 존재성 체크
        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_csv}")
        if not self.classes_json.exists():
            raise FileNotFoundError(f"classes json not found: {self.classes_json}")

        # 매니페스트 로드 및 스키마 검증
        self.df = pd.read_csv(self.manifest_csv)
        required_cols = ["image_path", "edi_code"]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"manifest_enriched.csv에 필수 컬럼이 없습니다: {missing_cols}"
            )

        # EDI 코드가 있는 샘플만 유지
        initial_count = len(self.df)
        self.df = self.df[self.df["edi_code"].notna() & (self.df["edi_code"] != "")]
        final_count = len(self.df)

        if final_count == 0:
            raise ValueError("매니페스트에 유효한 EDI 코드가 있는 샘플이 없습니다")

        print(
            f"📊 Dataset loaded: {initial_count} -> {final_count} samples with valid EDI codes"
        )

        # 클래스 맵 로드
        with open(self.classes_json, "r", encoding="utf-8") as f:
            self.clsmap = json.load(f)

        print(f"📋 Class map loaded: {len(self.clsmap)} EDI classes")

        # 빠른 샘플링을 위한 인덱스 준비
        self.samples = []
        unknown_edi_count = 0

        for _, row in self.df.iterrows():
            img_path = row["image_path"]
            edi_code = str(row["edi_code"]).strip()

            if edi_code in self.clsmap:
                self.samples.append((img_path, edi_code))
            else:
                unknown_edi_count += 1

        if unknown_edi_count > 0:
            print(f"⚠️  {unknown_edi_count} samples with unknown EDI codes (excluded)")

        if len(self.samples) == 0:
            raise ValueError("클래스 맵에 매칭되는 EDI 코드가 있는 샘플이 없습니다")

        print(f"✅ Final dataset: {len(self.samples)} samples ready for training")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, edi_code = self.samples[idx]
        img_path = Path(img_path)

        # 파일 존재성 체크 (옵션)
        if self.require_exists and not img_path.exists():
            raise FileNotFoundError(f"image missing: {img_path}")

        # 이미지 로드
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {img_path}: {e}")

        # 전처리 적용
        if self.transform:
            im = self.transform(im)

        # 클래스 ID 매핑
        class_id = int(self.clsmap[edi_code])

        return im, class_id

    def get_class_info(self):
        """클래스 정보 반환"""
        return {
            "num_classes": len(self.clsmap),
            "class_map": self.clsmap.copy(),
            "samples_per_class": self._count_samples_per_class(),
        }

    def _count_samples_per_class(self):
        """클래스별 샘플 수 계산"""
        class_counts = {}
        for _, edi_code in self.samples:
            class_id = self.clsmap[edi_code]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        return class_counts
