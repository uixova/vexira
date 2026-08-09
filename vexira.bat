@echo off
REM Vexira - tek tikla ceviri menusu  (Windows)
REM Cift tikla calisir; .bat zaten kendi konsol penceresini acar.
REM PowerShell tercih edersen: vexira.ps1

chcp 65001 >nul
cd /d "%~dp0"
title Vexira - TR / EN ceviri

set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY ( where py >nul 2>&1 && set "PY=py" )
if not defined PY ( where python3 >nul 2>&1 && set "PY=python3" )

if not defined PY (
  echo.
  echo   Python bulunamadi.
  echo   Kur: https://www.python.org/downloads/
  echo   Kurarken "Add Python to PATH" kutusunu ISARETLE.
  echo.
  pause
  exit /b 1
)

%PY% menu.py %*

REM Pencere hemen kapanmasin - son cikti okunabilsin.
echo.
pause
