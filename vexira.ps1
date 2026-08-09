# Vexira — tek tıkla çeviri menüsü  (Windows PowerShell)
#
# ÇİFT TIKLAMA: .ps1 dosyaları çift tıklanınca ÇALIŞMAZ, Not Defteri'nde açılır —
# Windows'un varsayılan davranışı bu. İki yol:
#   1) vexira.bat kullan (çift tıkla çalışır, önerilen)
#   2) bu dosyaya sağ tık > "Run with PowerShell"
#
# ExecutionPolicy engellerse:
#     powershell -ExecutionPolicy Bypass -File vexira.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Türkçe karakterler bozulmasın
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "Vexira — TR / EN çeviri"

$py = $null
foreach ($cand in @("python", "py", "python3")) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) { $py = $cand; break }
}

if (-not $py) {
    Write-Host ""
    Write-Host "  Python bulunamadı." -ForegroundColor Red
    Write-Host "  Kur: https://www.python.org/downloads/"
    Write-Host '  Kurarken "Add Python to PATH" kutusunu İŞARETLE.'
    Write-Host ""
    Read-Host "  Kapatmak için Enter"
    exit 1
}

& $py menu.py @args

# Pencere hemen kapanmasın — son çıktı okunabilsin.
Write-Host ""
Read-Host "  Kapatmak için Enter"
