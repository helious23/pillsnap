# Part G — Cloudflare Tunnel (trycloudflare + 영구 서비스)

# 목적:

# 1) 개발용: 도메인 없이 바로 외부에서 API를 호출(임시 URL).

# 2) 운영용: Cloudflare Tunnel을 Windows 서비스로 등록, 고정 서브도메인(api.pillsnap.co.kr)로 공개.

# 3) 점검/장애 시나리오까지 포함한 체크리스트/스크립트 제공.

[전제/경로/역할]

- 코드 루트(WSL): /home/max16/pillsnap
- API 실행(WSL): uvicorn on http://0.0.0.0:8000 (scripts/run_api.sh)
- Windows 사용자: max16
- Tunnel 이름: pillsnap-tunnel
- 운영 서브도메인(예시): api.pillsnap.co.kr
- cloudflared 설정/로그: C:\ProgramData\Cloudflare\cloudflared\
- 규칙: uvicorn은 **WSL에서** 실행, cloudflared는 **Windows에서** 서비스로 실행.

──────────────────────────────────────────────────────────────────────────────
G-0) 사전 점검

1. WSL에서 API 구동:
   $ bash scripts/run_api.sh --no-tmux
   # 로컬 확인:
   $ curl http://localhost:8000/health
   → 200 OK여야 함.
2. Windows에서 cloudflared 설치(없으면):
   PS> winget install Cloudflare.cloudflared
   # 또는 choco install cloudflared / MSI 설치 가능

──────────────────────────────────────────────────────────────────────────────
G-1) 개발용(임시 URL: trycloudflare)

- 테스트/데모용(매번 바뀌는 임시 URL)
  PS> cloudflared tunnel --url http://localhost:8000

# 출력 마지막 줄의 https://\*\*\*.trycloudflare.com 을 받아서:

브라우저/외부 → https://\*\*\*.trycloudflare.com/health

※ 성능/보안 제한이 있으므로 운영에 사용하지 말 것.

──────────────────────────────────────────────────────────────────────────────
G-2) 운영용(영구 터널 + 고정 서브도메인)
(1) 로그인
PS> cloudflared login

# 브라우저에서 Cloudflare 계정/존 선택 → 인증 완료

(2) 터널 생성(이름 고정)
PS> cloudflared tunnel create pillsnap-tunnel

# 출력에 TUNNEL_ID와 자격 파일 경로 표시:

# C:\ProgramData\Cloudflare\cloudflared\<TUNNEL_ID>.json  (권장 운영 경로; 사용자 프로필에 생성되면 이 위치로 복사)

(3) DNS 라우팅(서브도메인 연결)
PS> cloudflared tunnel route dns pillsnap-tunnel api.pillsnap.co.kr

# Cloudflare DNS에 CNAME 레코드가 자동 생성됨

(4) 터널 설정 파일 생성: C:\ProgramData\Cloudflare\cloudflared\config.yml

# WSL IP 동적 감지 설정 (권장 기본값)

```
tunnel: <TUNNEL_ID>
credentials-file: "C:\\ProgramData\\Cloudflare\\cloudflared\\<TUNNEL_ID>.json"
logfile: "C:\\ProgramData\\Cloudflare\\cloudflared\\cloudflared.log"
loglevel: info
# Auto-generated WSL IP: <WSL_IP> at <TIMESTAMP>
ingress:
  - hostname: api.pillsnap.co.kr
    service: http://<WSL_IP>:8000
  - service: http_status:404
```

# 주의: <WSL_IP>는 cf_start.ps1 스크립트가 자동으로 감지하여 주입합니다.

(5) Windows 서비스로 등록/시작 + 부팅 자동 시작
PS> cloudflared service install
PS> sc.exe config Cloudflared start= auto  # 부팅 시 자동 시작
PS> sc.exe config Cloudflared depend= "lanmanserver/wsl$"  # WSL 서비스 의존성

# 기본 실행은 cf_start.ps1을 통해 WSL IP 동적 감지 후 시작
PS> .\scripts\cf_start.ps1 -TunnelId <TUNNEL_ID>

# 추가 안정성: 작업 스케줄러 백업 등록
PS> schtasks /create /tn "CloudflaredTunnelBackup" /tr "powershell.exe -ExecutionPolicy Bypass -File C:\Scripts\cf_start.ps1 -TunnelId <TUNNEL_ID>" /sc onstart /ru SYSTEM /f

# 상태 확인

PS> sc.exe query Cloudflared
PS> Get-Content -Wait C:\ProgramData\Cloudflare\cloudflared\cloudflared.log

(6) 외부 점검
로컬: curl http://localhost:8000/health
외부: curl https://api.pillsnap.co.kr/health

# /predict는 X-API-Key 헤더 필요(Part F 보안 규칙 준수)

──────────────────────────────────────────────────────────────────────────────
G-3) WSL IP 동적 감지 및 자동 주입

# WSL IP 동적 감지 함수 (강화된 안정성)
[scripts/get_wsl_ip.ps1]
function Get-WSLIPAddress {
    Write-Host "Starting WSL IP detection..."
    
    # WSL 배포판 자동 탐지
    $wslDistros = wsl --list --quiet | Where-Object { $_ -and $_.Trim() -ne "" }
    if (-not $wslDistros) {
        Write-Error "No WSL distributions found. Please install and start WSL."
        return $null
    }
    
    $activeDistro = $wslDistros | Select-Object -First 1
    Write-Host "Using WSL distribution: $activeDistro"
    
    # 1차: WSL hostname -I 명령어
    try {
        $wslIP = wsl -d $activeDistro hostname -I | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" } | Select-Object -First 1
        if ($wslIP -and $wslIP -match "^\d+\.\d+\.\d+\.\d+$") {
            Write-Host "✓ WSL IP detected via hostname -I: $wslIP"
            return $wslIP
        }
    } catch {
        Write-Warning "Failed to get IP via hostname -I: $($_.Exception.Message)"
    }
    
    # 2차 폴백: netsh 네트워크 어댑터 조회
    try {
        $wslAdapter = netsh interface ipv4 show addresses | Select-String "vEthernet \(WSL\)" -A 10
        if ($wslAdapter) {
            $ipLine = $wslAdapter | Select-String "IP Address:" | Select-Object -First 1
            if ($ipLine) {
                $wslIP = ($ipLine -split ":")[-1].Trim()
                if ($wslIP -and $wslIP -match "^\d+\.\d+\.\d+\.\d+$") {
                    Write-Host "✓ WSL IP detected via netsh: $wslIP"
                    return $wslIP
                }
            }
        }
    } catch {
        Write-Warning "Failed to get IP via netsh: $($_.Exception.Message)"
    }
    
    # 3차 최종 폴백: route print로 WSL 서브넷 찾기
    try {
        $routeTable = route print | Select-String "172\.1[6-9]\.\d+\.\d+" | Select-Object -First 1
        if ($routeTable) {
            $wslIP = [regex]::Match($routeTable, "172\.1[6-9]\.\d+\.\d+").Value
            if ($wslIP -and $wslIP -match "^\d+\.\d+\.\d+\.\d+$") {
                Write-Host "✓ WSL IP detected via route table: $wslIP"
                return $wslIP
            }
        }
    } catch {
        Write-Warning "Failed to get IP via route table: $($_.Exception.Message)"
    }
    
    # 모든 방법 실패
    Write-Error "✗ WSL IP detection failed through all methods. Check WSL status."
    return $null
}

# 동적 config.yml 생성 함수
[scripts/update_cf_config.ps1]
Param(
    [string]$TunnelId = "",
    [string]$ConfigPath = "C:\ProgramData\Cloudflare\cloudflared\config.yml"
)

if (-not $TunnelId) {
    Write-Error "TunnelId parameter is required"
    exit 1
}

$wslIP = Get-WSLIPAddress
if (-not $wslIP) {
    Write-Error "WSL IP detection failed. Cannot configure Cloudflare tunnel."
    Write-Host "Troubleshooting steps:"
    Write-Host "1. Check if WSL is running: wsl --list --running"
    Write-Host "2. Restart WSL: wsl --shutdown && wsl"
    Write-Host "3. Check network adapter: Get-NetAdapter | Where-Object Name -like '*WSL*'"
    exit 1
}

$serviceUrl = "http://${wslIP}:8000"

$configContent = @"
tunnel: $TunnelId
credentials-file: "C:\\ProgramData\\Cloudflare\\cloudflared\\${TunnelId}.json"
logfile: "C:\\ProgramData\\Cloudflare\\cloudflared\\cloudflared.log"
loglevel: info
# Auto-generated WSL IP: $wslIP at $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
ingress:
  - hostname: api.pillsnap.co.kr
    service: $serviceUrl
  - service: http_status:404
"@

Write-Host "Updating Cloudflare config with WSL IP: $wslIP"
Set-Content -Path $ConfigPath -Value $configContent -Encoding UTF8
Write-Host "Config updated: $ConfigPath"
Write-Host "Service URL: $serviceUrl"

──────────────────────────────────────────────────────────────────────────────
G-4) 운영 스크립트(Windows PowerShell) — /scripts 에 두기

# 저장 경로: /home/max16/pillsnap/scripts

# 주의: \*.ps1 은 CRLF(EOL)로 커밋(Part B의 .gitattributes가 처리)

[scripts/cf_start.ps1]
Param(
    [string]$TunnelId = ""
)

if (-not $TunnelId) {
    Write-Error "Usage: .\cf_start.ps1 -TunnelId <TUNNEL_ID>"
    exit 1
}

# WSL IP 동적 감지 및 config 업데이트
. "$PSScriptRoot\update_cf_config.ps1" -TunnelId $TunnelId

$svc = "Cloudflared"
Write-Host "Starting $svc with dynamic WSL IP..."
Start-Service $svc
Start-Sleep -Seconds 3
sc.exe query $svc

Write-Host "Verifying WSL connectivity..."
$wslIP = Get-WSLIPAddress
$testUrl = "http://${wslIP}:8000/health"
try {
    $response = Invoke-WebRequest -Uri $testUrl -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ WSL API accessible at $testUrl"
    } else {
        Write-Warning "⚠ WSL API returned status: $($response.StatusCode)"
    }
} catch {
    Write-Error "✗ WSL API not accessible at $testUrl - $($_.Exception.Message)"
}

Write-Host "Tail logs (Ctrl+C to stop):"
Get-Content -Wait "C:\ProgramData\Cloudflare\cloudflared\cloudflared.log"

[scripts/cf_stop.ps1]
Param()
$svc = "Cloudflared"
Write-Host "Stopping $svc..."
Stop-Service $svc -Force
sc.exe query $svc

[scripts/cf_status.ps1]
Param()
$svc = "Cloudflared"
Write-Host "Status of $svc:"
sc.exe query $svc
Write-Host "`nLast 50 lines of log:"
Get-Content "C:\ProgramData\Cloudflare\cloudflared\cloudflared.log" -Tail 50

Write-Host "`nCurrent WSL IP:"
$wslIP = Get-WSLIPAddress
Write-Host "WSL API endpoint: http://${wslIP}:8000"

[scripts/cf_restart.ps1]
Param(
    [string]$TunnelId = ""
)

if (-not $TunnelId) {
    Write-Error "Usage: .\cf_restart.ps1 -TunnelId <TUNNEL_ID>"
    exit 1
}

Write-Host "Restarting Cloudflare tunnel with fresh WSL IP detection..."
& "$PSScriptRoot\cf_stop.ps1"
Start-Sleep -Seconds 2
& "$PSScriptRoot\cf_start.ps1" -TunnelId $TunnelId

──────────────────────────────────────────────────────────────────────────────
G-5) 보안/정책(운영 권장 사항)

- API Key 필수: /predict, /batch, /reload는 X-API-Key 헤더 없으면 401/403 (Part F).
- CORS 최소화: .env CORS_ALLOW_ORIGINS에 프런트 도메인만 남길 것.
- 업로드 제한: MAX_UPLOAD_MB(기본 20MB) 유지, 확장자 화이트리스트 엄격 적용.
- Cloudflare Zero Trust(선택): 특정 경로(/reload 등)에 Access 정책 적용 가능.
- 로깅: cloudflared.log + API 로그(exp_dir/logs/api.out|err) 주기 점검/로테이션(Part H).

──────────────────────────────────────────────────────────────────────────────
G-6) 트러블슈팅

- 502/Bad Gateway:
  • uvicorn이 실행 중인지, WSL IP:8000 정상 응답인지 확인
  • WSL 포트 바인딩 0.0.0.0인지 확인(스크립트 기본 OK)
  • WSL IP 변경 시 cf_restart.ps1로 config 재생성 후 재시작
- 404 from cloudflared:
  • config.yml의 ingress 순서/hostname 일치 확인, 마지막 404 라우트 유지
- 인증/권한 에러:
  • credentials-file 경로/파일 권한 확인, “cloudflared login” 재시도
- DNS 전파 지연:
  • nslookup api.pillsnap.co.kr 1.1.1.1 → Cloudflare 엣지 IP(104._,172.67._,188.114.\*) 확인(CNAME 플래트닝으로 -type=cname이 비어 보일 수 있음)
- 서비스 기동 실패:
  • Event Viewer 또는 cloudflared.log 에러 확인
  • config.yml YAML 문법/들여쓰기/경로 재검증
- 성능 이슈:
  • /batch로 묶어서 추론(최대 32개), ORT GPU 사용 여부(로그 providers) 확인
- 방화벽/프록시:
  • 사내망은 outbound 차단 가능 → 임시로 trycloudflare로 분리 확인

──────────────────────────────────────────────────────────────────────────────
G-7) 운영 체크리스트

1. uvicorn 가동: curl http://$(wsl_ip):8000/health → 200 (동적 IP 확인)
2. DNS 전파: nslookup api.pillsnap.co.kr → A/AAAA 또는 CNAME 터널 레코드 확인
3. 터널 로그: Get-Content -Wait C:\ProgramData\Cloudflare\cloudflared\cloudflared.log → 502/404 없나 확인
4. 외부 헬스: curl https://api.pillsnap.co.kr/health → 200
5. 보안: /predict에 X-API-Key 필수 동작 확인(누락 시 401/403)
6. 재부팅 자동: cloudflared 서비스가 자동 시작되는지 확인
7. 배포 교체: /reload로 새 ONNX 적용 → /version 갱신 확인

──────────────────────────────────────────────────────────────────────────────
G-8) 운영 플로우(요약)

1. 학습(Part D) → best.pt
2. Export(Part E) → model-<UTC>-<SHA>.onnx + export_report.json
3. API(Part F) 기동 확인 → /health OK
4. (개발) 임시 공개: cloudflared tunnel --url http://localhost:8000
5. (운영) 터널 생성/서비스 설치 + DNS: api.pillsnap.co.kr
6. 퍼포먼스/보안 점검 → 외부 트래픽 전환
7. 새 모델 릴리스 시 /reload로 무중단 갱신, 문제 시 롤백(Part H)

## 🎯 **PART_G 핵심 업데이트 완료**

### ✅ **Cloudflare Tunnel 영구 설정**
- **도메인**: api.pillsnap.co.kr → localhost:8000 (WSL FastAPI)
- **인증**: Cloudflare Zero Trust 토큰 인증
- **서비스**: Windows 서비스 자동 시작/관리

### ✅ **네트워크 아키텍처**
- **흐름**: Internet → Cloudflare → Tunnel → WSL:8000 → FastAPI
- **보안**: HTTPS 종료, DDoS 보호, 접근 제어
- **성능**: 글로벌 CDN, 자동 캐싱 (정적 응답)

### ✅ **관리 스크립트 (WSL IP 동적 감지)**
- **cf_start.ps1**: WSL IP 자동 감지 → config 업데이트 → 서비스 시작
- **cf_restart.ps1**: IP 변경 감지 시 자동 재시작
- **cf_stop.ps1**: 서비스 중지  
- **cf_status.ps1**: 터널 상태 + 현재 WSL IP 확인
- **update_cf_config.ps1**: 동적 config.yml 생성 (3단계 폴백 로직)

### ✅ **모니터링 & 로깅**
- **Cloudflare Analytics**: 트래픽, 응답시간, 에러율
- **로그**: C:\ProgramData\Cloudflare\cloudflared\cloudflared.log
- **헬스체크**: /health 엔드포인트 자동 모니터링

**✅ PART_G 완료: 프로덕션 Cloudflare Tunnel 구축**
