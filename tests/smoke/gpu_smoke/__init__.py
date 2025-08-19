"""
GPU Smoke Tests for RTX 5080 (sm_120) - PyTorch 2.7.0+cu128

이 모듈은 RTX 5080에서 PyTorch 2.7.0+cu128의 정상 동작을 검증합니다.
- GPU_A: 순수 PyTorch 합성 데이터 테스트
- GPU_B: 실데이터 + PillsnapClsDataset 테스트  
- GPU_Stage2: pillsnap.stage2.train_cls GPU 모드 테스트
"""