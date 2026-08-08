#!/usr/bin/env bash
# Vexira — tek tıkla çeviri menüsü  (Linux / macOS)
#
# Çift tıkla ya da:  ./vexira.sh
# Windows için: vexira.bat  veya  vexira.ps1
#
# Modeli her seçimde yeniden yüklemez; tek Python süreci menüyü de yönetir.
# Model yükleme ~2 sn sürüyor, her işlemde tekrarlamak menüyü kullanılmaz yapardı.

cd "$(dirname "$(readlink -f "$0")")" || exit 1

PY=python3
command -v $PY >/dev/null 2>&1 || PY=python
command -v $PY >/dev/null 2>&1 || {
  echo "Python bulunamadı. Kur:  https://python.org"; read -rp "Enter..."; exit 1; }

exec $PY menu.py "$@"
