# Part H — 운영 팁 & 배포 운영 가이드

# (로그·백업·롤백·모니터링·성능·보안·스케일링·자동화)

[전제/경로]

- 코드 루트(WSL): /home/max16/pillsnap
- venv(WSL): $HOME/pillsnap/.venv
- 실험 디렉토리: **/home/max16/ssd_pillsnap/exp/exp01** (SSD 이전 완룼)
- API 포트: 8000 (uvicorn, Part F)
- Cloudflared(Windows): C:\ProgramData\Cloudflare\cloudflared\ (Part G)

[이 파트에서 구현/수정할 항목]

1. 스크립트(scripts/)
   - maintenance.sh # 로그/아티팩트 정리, 디스크 리포트
   - backup_release.sh # 릴리스 아카이브 생성(.tar.gz + sha256 + manifest)
   - reload_model.sh # /reload로 모델 무중단 교체
   - perf_bench_infer.py # Torch vs ONNX 추론 벤치(배치/스레드/EP)
   - ort_optimize.py # onnxruntime 프로바이더/옵션 탐색·추천
   - quantize_dynamic.py # ONNX 동적 양자화(INT8) + 정확도 비교
   - health_watch.ps1 # (Windows) 외부/내부 헬스체크, 실패 시 cloudflared 조치
   - rotate_logs.ps1 # (Windows) cloudflared.log 로테이션
2. 문서(docs/)
   - OPS_CHECKLIST.md # 일/주/월 점검표
   - ROLLBACK.md # 롤백 절차(스텝바이스텝)
   - SECURITY.md # API Key/CORS/Zero Trust/비밀관리
   - PERFORMANCE.md # ORT/Torch 튜닝/스레드/그래프옵트/양자화 가이드

──────────────────────────────────────────────────────────────────────────────
H-1) 로깅·보존·로테이션 정책

[WSL/API 로그]

- 위치: **/home/max16/ssd_pillsnap/exp/exp01/logs/** (SSD 저장)
  • train.out|err, api.out|err, export.out|err, perf_bench.json 등
- 보존: 7일↑ gzip 압축 → logs/archive/ 로 이동, 30일↑ 삭제(필요 시 변경)

[Windows/cloudflared 로그]

- 위치: C:\ProgramData\Cloudflare\cloudflared\cloudflared.log
- rotate_logs.ps1 요구사항:
  1. 파일 크기 ≥ 100MB 또는 7일 경과 시 cloudflared.log.YYYYMMDD.gz로 회전
  2. 최근 10개만 보존, 나머지 삭제
  3. 권한 문제/열림 상태일 때 안전 처리(임시 파일로 복사 후 원본 truncate)

[maintenance.sh 요구사항]

- set -euo pipefail
- exp_dir 하위:
  ```
  1. logs/*.out|*.err 7일↑ gzip → logs/archive/
  2. logs/archive/*.gz 30일↑ 삭제
  3. checkpoints/ 의 last_*.pt 14일↑ 정리(best.pt는 보존)
  4. export/*.onnx 5개 초과 시 오래된 것부터 제거(최신/심볼릭 유지)
  5. df -h / /mnt/data 결과를 reports/disk_YYYYMMDD.txt로 저장
  ```
- 실행 예:
  $ bash scripts/deployment/maintenance.sh

──────────────────────────────────────────────────────────────────────────────
H-2) 백업·릴리스 아카이브(배포 산출물 표준화)

[backup_release.sh 요구사항]

- set -euo pipefail
- 입력:
  • onnx 경로(선택). 없으면 export/ 최신 onnx 자동 탐색(latest.onnx → realpath).
- 산출물 디렉토리: {exp_dir}/releases/
- 포함 파일:
  • config.yaml, .env.example, requirements.txt
  • export/\*.onnx (또는 지정 onnx 1개만—옵션 스위치 제공)
  • export/export_report.json, reports/metrics.json(최신)
  • VERSION.txt (UTC+git SHA), MANIFEST.json (파일 목록/크기/sha256)
- 파일명: release-<UTC>-<sha|nogit>.tar.gz
- sha256sum: release-....tar.gz.sha256 생성
- 출력 경로 echo

[복구 가이드(문서에도 삽입)]

- 아카이브 풀기 → MODEL_PATH를 해당 onnx로 지정하거나 /reload로 교체 → /version 확인

──────────────────────────────────────────────────────────────────────────────
H-3) 운영 중 무중단 교체(/reload) & 롤백

[reload_model.sh 요구사항]

- 인자: --path /mnt/data/exp/exp01/export/model-....onnx
- .env에서 API_KEY 로드
- 호출: POST /reload {"model_path":"..."} → 성공 시 /version 확인
- 실패 시:
  • 응답 본문/상태코드 출력 → 이전 모델 유지
  • 힌트: 경로 오타/권한/손상 파일/클래스 불일치

[롤백 플로우(ROLLBACK.md에도 동일 서술)]

1. 안정판 onnx 경로 확인(릴리스 보관 또는 export/직전 파일)
2. reload_model.sh --path <안정판.onnx>
3. /version으로 교체 확인
4. 문제 지속 시 cloudflared 일시 중단(또는 방화벽 레벨) 후 점검

──────────────────────────────────────────────────────────────────────────────
H-4) 헬스체크·모니터링(간단→확장)

[health_watch.ps1 요구사항(Windows)]

- 매 1분 내부/외부 체크:
  • 내부: http://localhost:8000/health
  • 외부: https://api.pillsnap.co.kr/health
- 2회 연속 실패 시 조치:
  1. cloudflared 서비스 재시작
  2. 여전히 실패 → 간단 알림(콘솔/Windows 이벤트 로그) 남기고 종료코드≠0
- 로그: C:\ProgramData\Cloudflare\cloudflared\health_watch.log 에 append

[확장 아이디어(선택)]

- GitHub Actions/서드파티 Uptime 로봇으로 외부 헬스 모니터링
- /metrics(프로메테우스) 노출은 범위 밖. 필요 시 별 문서에서 확장

──────────────────────────────────────────────────────────────────────────────
H-5) 추론 성능 벤치·최적화(ORT/Torch)

[perf_bench_infer.py 요구사항]

- 입력:
  • --engine {onnx,torch,both}, --model PATH, --inputs "glob", --batch {1,2,4,8,16,32}
  • --threads {0|N} (ORT CPU 경로용), --providers "CUDA,CPU" (콤마)
- 출력:
  • 각 배치에 대한 latency(ms)/throughput(img/s) 표 및 JSON({exp_dir}/reports/bench_UTC.json)
- 규칙:
  • 전처리(src.infer.preprocess) 재사용
  • Torch는 eval+no_grad, ONNX는 select_onnx_providers 정책(Part E)
  • 결과 요약을 콘솔 테이블로 표시

[ort_optimize.py 요구사항]

- 여러 세션 옵션/EP 조합 벤치:
  • providers: [["CUDA"],["CUDA","CPU"],["CPU"]]
  • graph_optimization_level: ORT_DISABLE_ALL|BASIC|EXTENDED|ALL
  • cudnn_conv_use_max_workspace: 0|1
- 최고 throughput 조합 추천 및 JSON 리포트({exp_dir}/reports/ort_bench.json)

[quantize_dynamic.py 요구사항]

- 입력: --model model.onnx, --out model.int8.onnx
- 방식: onnxruntime.quantization.quantize_dynamic(weight_type=QInt8)
- 비교:
  • perf_bench_infer.py를 호출해 FP32 vs INT8 throughput 비교
  • 정확도: val 샘플 128장 기준 Top-1 하락률 보고(허용 임계 예: ≤0.5%—config로 조절)

──────────────────────────────────────────────────────────────────────────────
H-6) 보안·비밀 관리

[API Key]

- .env의 API_KEY는 최소 32바이트 난수(예: `openssl rand -hex 32`)
- 교체 절차:
  1. .env 갱신 → uvicorn 재시작
  2. 클라이언트 헤더 업데이트(X-API-Key)
- 로그에 Key 노출 금지(마스킹)

[CORS]

- 운영에서는 CORS_ALLOW_ORIGINS를 프런트 도메인만 남김
- 프리플라이트/실패 로그를 주기 점검

[업로드 제한·검증]

- MAX_UPLOAD_MB=20 기본 유지(대용량은 /batch로 나눔)
- 확장자·Content-Type 화이트리스트 강제

[Cloudflare Zero Trust(선택)]

- /reload, /batch 등 민감 경로에 Access 정책 적용 가능(전자메일/그룹 제한)

──────────────────────────────────────────────────────────────────────────────
H-7) 스케일링 전략

[수직 확장]

- 더 큰 GPU(메모리↑), ORT CUDA EP 최적화
- img_size/batch 조절로 지연/처리량 균형

[수평 확장]

- 단일 GPU = uvicorn workers=1 인스턴스 1개 원칙
- 여러 인스턴스(여러 WSL/머신) 실행 후 Cloudflare Load Balancer로 트래픽 분산
- 모델 파일은 공용 저장소(공유 마운트 or 릴리스 패키지 배포)로 동기화

[안정 장치]

- /reload는 인스턴스별로 순차 적용(애저블 스크립트/간단 루프)

──────────────────────────────────────────────────────────────────────────────
H-8) 자동화(스케줄러)

[WSL(cron)]

- 매일 02:00 유지보수:
  $ crontab -e
  0 2 \* \* \* bash /home/max16/pillsnap/scripts/maintenance.sh >> /mnt/data/exp/exp01/logs/cron_maint.out 2>&1

[Windows 작업 스케줄러]

- health_watch.ps1: 매 1분
- rotate_logs.ps1: 매일 03:00
- cloudflared 서비스 자동 시작은 Part G의 service install로 처리됨

──────────────────────────────────────────────────────────────────────────────
H-9) 문서 템플릿(요구 내용)

[docs/OPS_CHECKLIST.md]

- 일간: /health·cloudflared.log·API 로그 에러 확인, 디스크 사용률, 최신 모델 버전 태그
- 주간: 로그 로테이션 결과, 백업 릴리스 생성 및 무작위 복원 테스트
- 월간: API Key 교체, 성능 벤치 재실행(perf_bench_infer.py), 보안 점검(CORS/업로드 정책)

[docs/ROLLBACK.md]

- 사전조건: 안정판 onnx 경로 확보
- 절차:
  1. reload_model.sh --path <안정판.onnx>
  2. /version으로 확인
  3. 필요 시 cloudflared 일시 중단→원인 파악→복구
- 체크: /predict 정상, 지표 역전 없음

[docs/SECURITY.md]

- API Key 정책·교체 주기·보관
- CORS 설정 원칙
- 파일 업로드 제한/화이트리스트
- Cloudflare Access(선택)의 적용 예

[docs/PERFORMANCE.md]

- ORT 프로바이더/옵션 권장 조합(벤치 결과 스냅샷)
- 배치 크기/스레드/입력 크기 트레이드오프
- INT8 양자화 가이드(허용 정확도 하락 한계)

──────────────────────────────────────────────────────────────────────────────
H-10) 검증·운영 점검 흐름

1. API 기동: curl http://localhost:8000/health → 200
2. 외부 헬스: curl https://api.pillsnap.co.kr/health → 200
3. 릴리스 생성: bash scripts/backup_release.sh → releases/…tar.gz + .sha256
4. 성능 점검: python scripts/perf_bench_infer.py --engine both --model <onnx> --inputs "/mnt/data/AIHub_576/val/**/*.jpg" --batch 1,2,4,8,16
5. 최적화: python scripts/ort_optimize.py → reports/ort_bench.json 확인
6. 양자화(선택): python scripts/quantize_dynamic.py --model <onnx> --out <int8.onnx> → 정확도/속도 비교
7. 롤백 리허설: scripts/reload_model.sh --path <직전 안정판.onnx> → /version 확인
8. 스케줄러: cron/작업 스케줄러 등록 및 최초 실행 로그 확인
9. 로그·보존: maintenance.sh / rotate_logs.ps1 수동 실행으로 정상 동작 체크

[Windows 서비스 표준 경로/명령]

- 설정/자격증명 위치(권장):
  - config.yml: C:\ProgramData\Cloudflare\cloudflared\config.yml
  - credentials: C:\ProgramData\Cloudflare\cloudflared\<TUNNEL_ID>.json
  - 로그: C:\ProgramData\Cloudflare\cloudflared\cloudflared.log
- 서비스 실행 명령 재설정(관리자 PowerShell):
  sc.exe config Cloudflared binPath= "\"C:\ProgramData\chocolatey\lib\cloudflared\tools\cloudflared.exe\" --config \"C:\ProgramData\Cloudflare\cloudflared\config.yml\" tunnel run <TUNNEL_ID>"
  sc.exe config Cloudflared start= auto
  sc.exe start Cloudflared
- 권한 강화(선택):
  icacls "C:\ProgramData\Cloudflare\cloudflared" /inheritance:r /grant:r "BUILTIN\Administrators":(OI)(CI)(F) "NT AUTHORITY\SYSTEM":(OI)(CI)(F) /T /C

## 🎯 **PART_H 핵심 업데이트 완료**

### ✅ **운영 자동화 시스템**
- **maintenance.sh**: 로그 로테이션, 체크포인트 정리, 디스크 리포트
- **backup_release.sh**: 모델 + 설정 아카이브 생성 (release-YYYYMMDD-SHA.tar.gz)
- **reload_model.sh**: 무중단 모델 교체 (/reload API 활용)

### ✅ **성능 벤치마킹 & 최적화**
- **perf_bench_infer.py**: PyTorch vs ONNX 성능 비교 (배치/스레드별)
- **ort_optimize.py**: ONNX Runtime 최적 설정 탐색
- **quantize_dynamic.py**: INT8 양자화 + 정확도 검증

### ✅ **모니터링 & 헬스체크**
- **health_watch.ps1**: 내부/외부 헬스체크 (Windows)
- **rotate_logs.ps1**: Cloudflare 로그 로테이션
- **자동 장애 대응**: 서비스 재시작, 알림 발송

### ✅ **문서 & 가이드**
- **OPS_CHECKLIST.md**: 일/주/월 운영 체크리스트
- **ROLLBACK.md**: 롤백 절차 가이드
- **SECURITY.md**: API 키, CORS, 업로드 보안
- **PERFORMANCE.md**: RTX 5080 최적화 가이드

### ✅ **배포 & 릴리스 관리**
- **아카이브**: /mnt/data/exp/exp01/releases/ 저장
- **버전 관리**: Git SHA + UTC 타임스탬프
- **무결성**: SHA256 체크섬 자동 생성

**✅ PART_H 완료: 프로덕션 운영 & 모니터링 시스템**
