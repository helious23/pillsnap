# cf_status.ps1
Param()
$svc = "cloudflared"
Write-Host "Status of $svc:"
sc.exe query $svc
Write-Host "`nLast 50 lines of log:"
Get-Content "$env:USERPROFILE\.cloudflared\cloudflared.log" -Tail 50
