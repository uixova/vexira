# Vexira — tek tıkla çeviri menüsü  (Windows PowerShell)
#
# Çalıştırma:  sağ tık > "Run with PowerShell"
# ExecutionPolicy engellerse:
#     powershell -ExecutionPolicy Bypass -File vexira.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Türkçe karakterler bozulmasın
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8

$py = $null
foreach ($cand in @("python", "py", "python3")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) { $py = $cand; break }
}
if (-not $py) {
    Write-Host "Python bulunamadı. Kur: https://python.org" -ForegroundColor Red
    Write-Host 'Kurarken "Add Python to PATH" kutusunu İŞARETLE.'
    Read-Host "Enter"
    exit 1
}

& $py menu.py @args

# Pencere hemen kapanmasın ki hata okunabilsin
if ($LASTEXITCODE -ne 0) { Read-Host "Enter" }
