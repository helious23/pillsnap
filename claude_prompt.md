 🎯 PillSnap ML 구현을 위한 단계별 명령 프롬프트

  📋 전체 구현 전략

  # PillSnap ML 프로젝트 구현 지시서

  ## 0. 초기 설정 및 환경 준비
  "프롬프트 PART_A부터 PART_H까지 모두 읽고, 프로젝트 구조와 요구사항을 완전히 이해한 후 진행해주세요."

  1. 프로젝트 디렉토리 구조 생성 (/home/max16/pillsnap)
  2. 가상환경 설정 스크립트 (scripts/bootstrap_venv.sh) 작성
  3. requirements.txt 생성 (필수 패키지만)
  4. .gitignore, .editorconfig 설정

  ## 1. Stage 1 최소 구현 (파이프라인 검증용)
  "Stage 1 (5,000장, 50클래스)에서 동작하는 최소 기능 구현을 목표로 합니다."

  ### 1.1 기본 설정 구현
  - PART_B의 config.yaml 생성 (Stage 1 설정만 활성화)
  - src/utils.py 작성 (load_config, set_seed, build_logger만 우선)

  ### 1.2 데이터 파이프라인 (간소화)
  - src/data.py 작성:
    - 단순 이미지 로더 (LMDB 없이 시작)
    - PillsnapClsDataset 클래스만 구현
    - safe_collate 함수
    - build_dataloaders() - Stage 1용 설정만

  ### 1.3 모델 및 학습 (단일 분류만)
  - src/models/classifier.py:
    - EfficientNetV2-L 래퍼 (timm 사용)
    - 50 클래스용 헤드
  - src/train.py:
    - train_classification_stage() 함수만 구현
    - 검증 루프 포함
    - 체크포인트 저장 (best.pt, last.pt)

  ### 1.4 Stage 1 검증
  - tests/test_stage_1.py:
    - 데이터 로딩 테스트
    - 1 epoch 학습 테스트
    - 체크포인트 저장/로드 테스트

  ## 2. Stage 2로 확장 (기본 성능 확보)
  "Stage 1이 정상 동작하면 Stage 2 (25,000장, 250클래스)로 확장합니다."

  ### 2.1 최적화 기능 추가
  - src/utils.py 확장:
    - auto_find_batch_size() 구현
    - enable_tf32(), try_torch_compile() 추가
    
  ### 2.2 RAM 최적화 적용
  - src/data.py 업데이트:
    - 간단한 메모리 캐싱 추가 (LMDB 아직 미적용)
    - num_workers 오토튜닝 적용

  ### 2.3 성능 모니터링
  - TensorBoard 통합
  - 학습 메트릭 로깅 강화

  ## 3. Two-Stage Pipeline 구현 (Stage 3 준비)
  "Stage 3 (100,000장, 1,000클래스)에서 필요한 검출 기능을 추가합니다."

  ### 3.1 검출 모델 통합
  - src/models/detector.py:
    - YOLOv11x 래퍼 구현
    - PillsnapDetDataset 클래스 추가

  ### 3.2 Interleaved 학습
  - src/train.py 수정:
    - interleaved 학습 전략 구현
    - detection/classification 교차 학습

  ### 3.3 ONNX Export
  - src/export.py:
    - 두 모델 각각 ONNX 변환
    - 동등성 검증

  ## 4. Stage 4 프로덕션 준비
  "Stage 4 (500,000장, 5,000클래스) 전체 기능 완성"

  ### 4.1 전체 최적화 적용
  - LMDB 데이터 변환
  - 전체 RAM 최적화 기능 활성화
  - torch.compile max-autotune 적용

  ### 4.2 API 서빙
  - src/api/app.py (FastAPI)
  - 추론 파이프라인 완성

  ### 4.3 최종 평가
  - final_test_evaluation() 실행
  - 성능 목표 달성 확인

  ---
  💻 실제 Claude Code 명령 예시

  Phase 1: 프로젝트 초기화

  "PART_A~H를 모두 읽고, Stage 1 (5,000장, 50클래스)에서 동작하는 최소 구현체를 만들어주세요.

  1. 먼저 프로젝트 구조를 생성하고 필요한 디렉토리를 만들어주세요.
  2. PART_B의 config.yaml을 생성하되, Stage 1 관련 설정만 활성화해주세요.
  3. src/utils.py에 기본 유틸리티 함수만 구현해주세요 (load_config, set_seed, build_logger).
  4. 복잡한 최적화는 나중에 추가할 예정이니 일단 동작하는 코드를 우선으로 해주세요."

  Phase 2: 데이터 파이프라인

  "Stage 1용 간단한 데이터 로더를 구현해주세요.

  1. src/data.py에 PillsnapClsDataset 클래스를 구현해주세요.
  2. LMDB나 복잡한 캐싱은 일단 제외하고, 기본 이미지 로딩만 구현해주세요.
  3. safe_collate 함수로 손상된 이미지 처리를 포함해주세요.
  4. build_dataloaders()는 Stage 1 설정(num_workers=8, prefetch_factor=4)만 사용해주세요."

  Phase 3: 모델 학습

  "Stage 1에서 분류 모델 학습이 가능하도록 구현해주세요.

  1. src/models/classifier.py에 EfficientNetV2-L 래퍼를 만들어주세요 (50 클래스).
  2. src/train.py에 train_classification_stage() 함수를 구현해주세요.
  3. 동적 배치 크기 조정은 일단 제외하고 고정 배치(32)로 시작해주세요.
  4. 1 epoch 학습 후 체크포인트가 저장되는지 확인해주세요."

  Phase 4: Stage별 점진적 확장

  "Stage 1이 정상 동작합니다. 이제 Stage 2 (25,000장, 250클래스)로 확장해주세요.

  1. config.yaml의 stage_overrides를 활용해 Stage 2 설정을 적용해주세요.
  2. auto_find_batch_size() 함수를 구현해 동적 배치 크기를 적용해주세요.
  3. 간단한 메모리 캐싱을 추가해주세요 (아직 LMDB는 불필요).
  4. Stage 2에서 목표 성능(60% accuracy)을 달성하는지 확인해주세요."

  Phase 5: Two-Stage 구현

  "Stage 3 준비를 위해 검출 모델과 interleaved 학습을 추가해주세요.

  1. YOLOv11x 검출 모델을 통합해주세요.
  2. PART_B의 interleaved 학습 전략(ratio [1,1])을 구현해주세요.
  3. detection과 classification이 번갈아 학습되는지 확인해주세요.
  4. 두 모델 모두 ONNX로 export되는지 테스트해주세요."

  Phase 6: 최종 최적화

  "Stage 4 (500,000장, 5,000클래스) 프로덕션을 위한 전체 최적화를 적용해주세요.

  1. PART_C의 LMDB + LRU 캐시 시스템을 구현해주세요.
  2. torch.compile을 max-autotune으로 설정하고 성능을 측정해주세요.
  3. final_test_evaluation()을 실행해 목표 달성 여부를 확인해주세요.
  4. FastAPI 서빙 코드를 완성해주세요."

  ---
  🎯 핵심 원칙

  1. Stage별 점진적 구현: Stage 1부터 시작해 검증 후 확장
  2. 동작 우선: 최적화보다 먼저 동작하는 코드 확보
  3. 테스트 주도: 각 Stage마다 검증 후 다음 단계 진행
  4. 명확한 목표: 각 단계의 성공 기준 명시
  5. 롤백 가능: 문제 발생 시 이전 Stage로 돌아갈 수 있도록