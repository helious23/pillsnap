# cf_stop.ps1
Param()
$svc = "cloudflared"
Write-Host "Stopping $svc..."
Stop-Service $svc -Force
sc.exe query $svc
