@echo off
REM Vexira - tek tikla ceviri menusu  (Windows, cift tikla calisir)
REM PowerShell tercih edersen: vexira.ps1

chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo Python bulunamadi. Kur: https://python.org
    echo Kurarken "Add Python to PATH" kutusunu ISARETLE.
    pause
    exit /b 1
  )
  py menu.py %*
) else (
  python menu.py %*
)

REM Cift tiklamada pencere kapanmasin ki hata mesaji okunabilsin.
if errorlevel 1 pause
