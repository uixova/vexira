#!/usr/bin/env bash
# Vexira — tek tıkla çeviri menüsü  (Linux / macOS)
#
# Çift tıkla ya da terminalden:  ./vexira.sh
# Windows için: vexira.bat  veya  vexira.ps1
#
# Modeli her seçimde yeniden yüklemez; tek Python süreci menüyü de yönetir.
# Model yükleme ~2 sn sürüyor, her işlemde tekrarlamak menüyü kullanılmaz yapardı.

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")" || exit 1

# --- ÇİFT TIKLAMA DESTEĞİ ---------------------------------------------------
# Dosya yöneticisinden çift tıklanınca script'in çıktısı terminale bağlı DEĞİL;
# menü girdi bekler ama kimse yazamaz ve pencere hiç açılmaz. Bu durumda
# kendimizi bir terminal emülatöründe yeniden başlatıyoruz.
#
# İKİSİ BİRDEN kopuksa yeniden başlatılır. Tek başına stdin ya da stdout
# bakmak yanlış sonuç verir:
#   ./vexira.sh            stdin ✓ stdout ✓  -> terminal, başlatma
#   printf 'q' | ./vexira  stdin ✗ stdout ✓  -> bilinçli boru, başlatma
#   ./vexira.sh > log      stdin ✓ stdout ✗  -> çıktı yönlendirme, başlatma
#   dosya yöneticisi       stdin ✗ stdout ✗  -> ÇİFT TIKLAMA, başlat
if [ ! -t 0 ] && [ ! -t 1 ]; then
  if [ "$(uname)" = "Darwin" ]; then
    open -a Terminal "$SELF"
    exit 0
  fi
  for term in x-terminal-emulator konsole gnome-terminal xfce4-terminal \
              kitty alacritty tilix mate-terminal lxterminal xterm; do
    if command -v "$term" >/dev/null 2>&1; then
      case "$term" in
        gnome-terminal) exec "$term" -- bash "$SELF" ;;
        konsole)        exec "$term" --hold -e bash "$SELF" ;;
        *)              exec "$term" -e bash "$SELF" ;;
      esac
    fi
  done
  echo "[!] Terminal emülatörü bulunamadı. Terminalden çalıştır:  bash vexira.sh"
  exit 1
fi

PY=""
for c in python3 python; do
  command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }
done

if [ -z "$PY" ]; then
  echo
  echo "  Python bulunamadı."
  echo "  Kur:  https://www.python.org/downloads/"
  echo
  # Otomatik KURMUYORUZ: sudo ister, dağıtıma göre değişir, kullanıcının
  # sistemine habersiz paket yükler. Komutu söylemek yeter.
  echo "  Arch:   sudo pacman -S python"
  echo "  Debian: sudo apt install python3 python3-pip"
  echo "  Fedora: sudo dnf install python3 python3-pip"
  echo
  read -r -p "  Kapatmak için Enter..." _
  exit 1
fi

"$PY" menu.py "$@"

# Çift tıklamayla açılan pencere hemen kapanmasın — son çıktı okunabilsin.
# "herhangi bir tuş" (read -n1 -s) her terminalde güvenilir değil;
# Enter beklemek her yerde çalışır.
echo
read -r -p "  Kapatmak için Enter..." _
