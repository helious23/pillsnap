# cf_start.ps1
Param()
$svc = "cloudflared"
Write-Host "Starting $svc..."
Start-Service $svc
Start-Sleep -Seconds 1
sc.exe query $svc
Write-Host "Tail logs (Ctrl+C to stop):"
Get-Content -Wait "$env:USERPROFILE\.cloudflared\cloudflared.log"
