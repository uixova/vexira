#!/usr/bin/env bash
# ============================================================================
# VEXIRA — GECE AFK EĞİTİM KOŞUCUSU
# ----------------------------------------------------------------------------
# Uyku/ekran kapanma ENGELLİ (systemd-inhibit). Ctrl+C ile temiz durur.
#
# İki not:
#   1. POWEROFF YOK — makine açık kalır, ertesi gece checkpoint'ten anında devam
#   2. DUR sentinel dosyası ile temiz durdurma
#
# Bu YEREL eğitim (sanity/debug). Tam eğitim uzak sunucuda.
#
# Kullanım:
#   ./afk_train.sh                       # 9 saat, tiny
#   ./afk_train.sh --preset main --max-hours 10
#   touch ~/ai-data/vexira/DUR           # temiz durdur
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")" || exit 1

POOL="${VEXIRA_POOL:-$HOME/ai-data/vexira}"
STOP_FILE="$POOL/DUR"
LOG="vexira_afk.log"

# Uyku engelleyici altında kendini yeniden çalıştır (bir kez)
if [ -z "${VEXIRA_INHIBITED:-}" ] && command -v systemd-inhibit >/dev/null 2>&1; then
  export VEXIRA_INHIBITED=1
  exec systemd-inhibit \
    --what=sleep:idle:handle-lid-switch \
    --who="Vexira AFK" --why="Gece egitimi (yerel sanity)" \
    bash "$0" "$@"
fi

notify() {
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -a "Vexira" "$1" "$2" 2>/dev/null || true
}

if [ -f "$STOP_FILE" ]; then
  echo "DUR dosyası var: $STOP_FILE — önce sil, sonra başlat."
  exit 1
fi

echo "============================================================"
echo "  VEXIRA AFK EĞİTİM"
echo "  log      : $(pwd)/$LOG"
echo "  durdurma : touch $STOP_FILE   (ya da Ctrl+C)"
echo "  uyku     : ENGELLİ"
echo "============================================================"
notify "Eğitim başladı" "PC'yi kapatma. Durdurmak için: touch $STOP_FILE"

trap 'echo; echo "[AFK] Ctrl+C — temiz duruluyor"; notify "Eğitim durduruldu" "Ctrl+C"; exit 0' INT TERM

start=$(date +%s)
python3 train_local.py "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
mins=$(( ($(date +%s) - start) / 60 ))

if [ -f "$STOP_FILE" ]; then
  echo "[AFK] DUR sentinel görüldü — durduruldu (${mins} dk)"
  notify "Eğitim durduruldu" "DUR dosyası, ${mins} dk koştu"
elif [ "$rc" -eq 0 ]; then
  echo "[AFK] bitti (${mins} dk)"
  notify "Eğitim bitti" "${mins} dk. Sıradaki: python evaluate.py"
else
  echo "[AFK] HATA rc=$rc (${mins} dk) — son satırlar:"
  tail -20 "$LOG"
  notify "Eğitim HATA verdi" "rc=$rc — $LOG dosyasına bak"
fi

# poweroff BİLİNÇLİ OLARAK YOK: checkpoint'ten devam etmek için makine açık kalsın
exit "$rc"
