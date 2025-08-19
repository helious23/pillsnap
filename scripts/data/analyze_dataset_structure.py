#!/usr/bin/env python3
"""
PillSnap 데이터 구조 스캔 스크립트
- 실제 압축 해제된 데이터 구조 분석
- 이미지/라벨 매칭 검증
- K-코드 매핑 테이블 생성
- 데이터 무결성 검사
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

# 프로젝트 루트 추가 (상위 디렉토리로 이동)
sys.path.insert(0, '/home/max16/pillsnap')

# src.utils 모듈에서 import (정리된 구조)
from src.utils import build_logger, load_config, ensure_dir


class PillSnapDataScanner:
    """
    PillSnap 데이터셋 구조 스캐너
    - ZIP 해제된 데이터 실제 구조 분석
    - 이미지-라벨 매칭 검증
    - EDI 코드 매핑 테이블 구축
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: 데이터셋 루트 경로 (/mnt/data/pillsnap_dataset)
        """
        self.data_root = Path(data_root)
        self.logger = build_logger("data_scanner", level="info")
        
        # 데이터 경로 설정
        self.train_images = self.data_root / "data" / "train" / "images"
        self.train_labels = self.data_root / "data" / "train" / "labels"
        self.val_images = self.data_root / "data" / "val" / "images"
        self.val_labels = self.data_root / "data" / "val" / "labels"
        
        # 분석 결과 저장용
        self.structure_info = {}
        self.k_code_mapping = {}
        self.edi_code_mapping = {}
        self.integrity_issues = []
        
    def scan_full_structure(self) -> Dict:
        """전체 데이터 구조 스캔 및 분석"""
        
        self.logger.step("데이터 구조 전체 스캔", "실제 압축 해제된 데이터 분석 시작")
        
        try:
            # 1) 기본 디렉토리 존재 확인
            self._verify_basic_structure()
            
            # 2) Single 약품 데이터 분석
            single_info = self._scan_single_data()
            
            # 3) Combination 약품 데이터 분석
            combo_info = self._scan_combination_data()
            
            # 4) Validation 데이터 분석
            val_info = self._scan_validation_data()
            
            # 5) K-코드 및 EDI 코드 매핑 구축
            self._build_code_mappings()
            
            # 6) 데이터 무결성 검사
            self._verify_data_integrity()
            
            # 7) 전체 결과 취합
            self.structure_info = {
                "data_root": str(self.data_root),
                "scan_timestamp": self.logger.timer_start("scan_complete"),
                "train_data": {
                    "single": single_info,
                    "combination": combo_info
                },
                "validation_data": val_info,
                "k_code_mapping": self.k_code_mapping,
                "edi_code_mapping": self.edi_code_mapping,
                "integrity_issues": self.integrity_issues,
                "summary": self._generate_summary()
            }
            
            self.logger.success("데이터 구조 스캔 완료")
            return self.structure_info
            
        except Exception as e:
            self.logger.failure(f"데이터 구조 스캔 실패: {e}")
            raise
    
    def _verify_basic_structure(self) -> None:
        """기본 디렉토리 구조 검증"""
        
        required_dirs = [
            self.train_images / "single",
            self.train_images / "combination", 
            self.train_labels / "single",
            self.train_labels / "combination"
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"필수 디렉토리 없음: {dir_path}")
                
        self.logger.info("✅ 기본 디렉토리 구조 검증 완료")
    
    def _scan_single_data(self) -> Dict:
        """Single 약품 데이터 분석"""
        
        self.logger.info("📂 Single 약품 데이터 분석 중...")
        
        single_info = {
            "ts_directories": [],  # TS_*_single 디렉토리들
            "tl_directories": [],  # TL_*_single 디렉토리들
            "k_codes": set(),      # 모든 K-코드들
            "total_images": 0,     # 총 이미지 수
            "total_labels": 0,     # 총 라벨 수
            "sample_analysis": {}  # 샘플 분석 결과
        }
        
        # TS_*_single 디렉토리 스캔
        single_img_dirs = list(self.train_images.glob("single/TS_*_single"))
        single_info["ts_directories"] = [d.name for d in sorted(single_img_dirs)]
        
        # TL_*_single 디렉토리 스캔  
        single_lbl_dirs = list(self.train_labels.glob("single/TL_*_single"))
        single_info["tl_directories"] = [d.name for d in sorted(single_lbl_dirs)]
        
        # 첫 번째 TS 디렉토리 상세 분석
        if single_img_dirs:
            sample_ts = single_img_dirs[0]
            k_code_dirs = list(sample_ts.glob("K-*"))
            
            for k_dir in k_code_dirs[:3]:  # 처음 3개만 샘플 분석
                k_code = k_dir.name
                single_info["k_codes"].add(k_code)
                
                # 이미지 파일 수 카운트
                image_files = list(k_dir.glob("*.png")) + list(k_dir.glob("*.jpg"))
                single_info["total_images"] += len(image_files)
                
                # 샘플 분석 저장
                if k_code not in single_info["sample_analysis"]:
                    single_info["sample_analysis"][k_code] = {
                        "image_count": len(image_files),
                        "sample_files": [f.name for f in image_files[:3]]
                    }
        
        # 대응되는 라벨 확인
        if single_lbl_dirs:
            sample_tl = single_lbl_dirs[0]
            k_json_dirs = list(sample_tl.glob("K-*_json"))
            
            for k_json_dir in k_json_dirs[:3]:
                k_code = k_json_dir.name.replace("_json", "")
                
                # JSON 파일 수 카운트
                json_files = list(k_json_dir.glob("*.json"))
                single_info["total_labels"] += len(json_files)
                
                # 샘플 JSON 분석
                if json_files and k_code in single_info["sample_analysis"]:
                    single_info["sample_analysis"][k_code]["label_count"] = len(json_files)
                    single_info["sample_analysis"][k_code]["sample_labels"] = [f.name for f in json_files[:3]]
        
        single_info["k_codes"] = list(single_info["k_codes"])  # set을 list로 변환
        
        self.logger.info(f"✅ Single 데이터: TS 디렉토리 {len(single_info['ts_directories'])}개, K-코드 {len(single_info['k_codes'])}개")
        return single_info
    
    def _scan_combination_data(self) -> Dict:
        """Combination 약품 데이터 분석"""
        
        self.logger.info("🔗 Combination 약품 데이터 분석 중...")
        
        combo_info = {
            "ts_directories": [],  # TS_*_combo 디렉토리들
            "tl_directories": [],  # TL_*_combo 디렉토리들
            "combo_k_codes": set(), # 조합 K-코드들 (K-xxx-yyy-zzz-www 형태)
            "total_images": 0,
            "total_labels": 0,
            "sample_analysis": {}
        }
        
        # TS_*_combo 디렉토리 스캔
        combo_img_dirs = list(self.train_images.glob("combination/TS_*_combo"))
        combo_info["ts_directories"] = [d.name for d in sorted(combo_img_dirs)]
        
        # TL_*_combo 디렉토리 스캔
        combo_lbl_dirs = list(self.train_labels.glob("combination/TL_*_combo"))
        combo_info["tl_directories"] = [d.name for d in sorted(combo_lbl_dirs)]
        
        # 첫 번째 TS_combo 디렉토리 상세 분석
        if combo_img_dirs:
            sample_ts = combo_img_dirs[0]
            combo_k_dirs = list(sample_ts.glob("K-*"))
            
            for combo_k_dir in combo_k_dirs[:3]:  # 처음 3개만 샘플 분석
                combo_k_code = combo_k_dir.name
                combo_info["combo_k_codes"].add(combo_k_code)
                
                # 조합 이미지 파일 수 카운트
                image_files = list(combo_k_dir.glob("*.png")) + list(combo_k_dir.glob("*.jpg"))
                combo_info["total_images"] += len(image_files)
                
                # 샘플 분석 저장
                combo_info["sample_analysis"][combo_k_code] = {
                    "image_count": len(image_files),
                    "sample_files": [f.name for f in image_files[:3]],
                    "individual_k_codes": combo_k_code.split('-')[1:]  # K- 제거 후 분리
                }
        
        combo_info["combo_k_codes"] = list(combo_info["combo_k_codes"])
        
        self.logger.info(f"✅ Combination 데이터: TS 디렉토리 {len(combo_info['ts_directories'])}개, 조합 K-코드 {len(combo_info['combo_k_codes'])}개")
        return combo_info
    
    def _scan_validation_data(self) -> Dict:
        """Validation 데이터 분석"""
        
        self.logger.info("🔍 Validation 데이터 분석 중...")
        
        val_info = {
            "exists": False,
            "single": {"vs_directories": [], "vl_directories": []},
            "combination": {"vs_directories": [], "vl_directories": []},
            "total_images": 0,
            "total_labels": 0
        }
        
        if self.val_images.exists():
            val_info["exists"] = True
            
            # Validation Single 데이터
            val_single_dirs = list(self.val_images.glob("single/VS_*_single"))
            val_info["single"]["vs_directories"] = [d.name for d in sorted(val_single_dirs)]
            
            val_single_lbl_dirs = list(self.val_labels.glob("single/VL_*_single"))
            val_info["single"]["vl_directories"] = [d.name for d in sorted(val_single_lbl_dirs)]
            
            # Validation Combination 데이터
            val_combo_dirs = list(self.val_images.glob("combination/VS_*_combo"))
            val_info["combination"]["vs_directories"] = [d.name for d in sorted(val_combo_dirs)]
            
            val_combo_lbl_dirs = list(self.val_labels.glob("combination/VL_*_combo"))
            val_info["combination"]["vl_directories"] = [d.name for d in sorted(val_combo_lbl_dirs)]
        
        self.logger.info(f"✅ Validation 데이터: 존재 여부 {val_info['exists']}")
        return val_info
    
    def _build_code_mappings(self) -> None:
        """K-코드 및 EDI 코드 매핑 구축"""
        
        self.logger.info("🗂️ K-코드 및 EDI 코드 매핑 구축 중...")
        
        # 샘플 JSON 파일에서 EDI 코드 추출
        sample_json_path = self.train_labels / "single" / "TL_1_single" / "K-000059_json" / "K-000059_0_0_0_0_60_000_200.json"
        
        if sample_json_path.exists():
            with open(sample_json_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
                
            if "images" in sample_data and sample_data["images"]:
                image_info = sample_data["images"][0]
                
                # K-코드 매핑 정보 추출
                k_code = image_info.get("dl_mapping_code", "")
                edi_code = image_info.get("di_edi_code", "")
                drug_name = image_info.get("dl_name", "")
                
                self.k_code_mapping[k_code] = {
                    "edi_code": edi_code,
                    "drug_name": drug_name,
                    "drug_shape": image_info.get("drug_shape", ""),
                    "color_class1": image_info.get("color_class1", ""),
                    "color_class2": image_info.get("color_class2", ""),
                    "print_front": image_info.get("print_front", ""),
                    "print_back": image_info.get("print_back", ""),
                    "company": image_info.get("dl_company", "")
                }
                
                # EDI 코드 역방향 매핑
                if edi_code:
                    self.edi_code_mapping[edi_code] = k_code
        
        self.logger.info(f"✅ 코드 매핑: K-코드 {len(self.k_code_mapping)}개, EDI 코드 {len(self.edi_code_mapping)}개")
    
    def _verify_data_integrity(self) -> None:
        """데이터 무결성 검사"""
        
        self.logger.info("🔒 데이터 무결성 검사 중...")
        
        # 샘플 이미지-라벨 매칭 검사
        sample_img_dir = self.train_images / "single" / "TS_1_single" / "K-000059"
        sample_lbl_dir = self.train_labels / "single" / "TL_1_single" / "K-000059_json"
        
        if sample_img_dir.exists() and sample_lbl_dir.exists():
            img_files = {f.stem for f in sample_img_dir.glob("*.png")}
            json_files = {f.stem for f in sample_lbl_dir.glob("*.json")}
            
            # 매칭되지 않는 파일들 찾기
            unmatched_images = img_files - json_files
            unmatched_labels = json_files - img_files
            
            if unmatched_images:
                self.integrity_issues.append(f"매칭되지 않는 이미지: {len(unmatched_images)}개")
            
            if unmatched_labels:
                self.integrity_issues.append(f"매칭되지 않는 라벨: {len(unmatched_labels)}개")
        
        if not self.integrity_issues:
            self.logger.success("데이터 무결성 검사 통과")
        else:
            self.logger.warning(f"무결성 이슈 {len(self.integrity_issues)}개 발견")
    
    def _generate_summary(self) -> Dict:
        """전체 분석 결과 요약 생성"""
        
        train_data = self.structure_info.get("train_data", {})
        
        summary = {
            "total_ts_single_dirs": len(train_data.get("single", {}).get("ts_directories", [])),
            "total_ts_combo_dirs": len(train_data.get("combination", {}).get("ts_directories", [])),
            "total_k_codes": len(train_data.get("single", {}).get("k_codes", [])),
            "total_combo_k_codes": len(train_data.get("combination", {}).get("combo_k_codes", [])),
            "estimated_total_images": train_data.get("single", {}).get("total_images", 0) + train_data.get("combination", {}).get("total_images", 0),
            "data_integrity_status": "OK" if not self.integrity_issues else "ISSUES_FOUND",
            "ready_for_processing": len(self.integrity_issues) == 0
        }
        
        return summary
    
    def save_analysis_report(self, output_path: str) -> None:
        """분석 결과를 JSON 파일로 저장"""
        
        output_file = Path(output_path)
        ensure_dir(output_file.parent)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.structure_info, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.success(f"분석 리포트 저장 완료: {output_file}")


def main():
    """메인 실행 함수"""
    
    # 설정 로딩
    config = load_config()
    data_root = config["data"]["root"]
    
    # 데이터 스캐너 생성 및 실행
    scanner = PillSnapDataScanner(data_root)
    
    try:
        # 전체 구조 스캔
        analysis_result = scanner.scan_full_structure()
        
        # 결과 출력
        summary = analysis_result["summary"]
        print("\n" + "="*60)
        print("📊 PillSnap 데이터 구조 분석 결과")
        print("="*60)
        print(f"📂 Single TS 디렉토리: {summary['total_ts_single_dirs']}개")
        print(f"🔗 Combination TS 디렉토리: {summary['total_ts_combo_dirs']}개")
        print(f"🏷️ 총 K-코드: {summary['total_k_codes']}개")
        print(f"🔗 총 조합 K-코드: {summary['total_combo_k_codes']}개")
        print(f"📷 예상 총 이미지: {summary['estimated_total_images']:,}개")
        print(f"🔒 데이터 무결성: {summary['data_integrity_status']}")
        print(f"✅ 처리 준비 상태: {summary['ready_for_processing']}")
        
        # 리포트 저장
        report_path = "/mnt/data/exp/exp01/reports/data_structure_analysis.json"
        scanner.save_analysis_report(report_path)
        
        if summary["ready_for_processing"]:
            print("\n🎉 데이터 구조 분석 완료! 데이터 파이프라인 구현을 진행할 수 있습니다.")
        else:
            print("\n⚠️ 데이터 무결성 이슈가 발견되었습니다. 확인이 필요합니다.")
        
    except Exception as e:
        print(f"\n❌ 데이터 구조 분석 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()