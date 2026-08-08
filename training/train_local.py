#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vexira yerel eğitim sarmalayıcısı — tek cihaz, debug ve gece AFK koşusu.

⚠️ Bu dosya AYRI bir eğitim döngüsü DEĞİL. train.py'yi çağırır, kendi
checkpoint şemasını yazmaz — yerel ve uzak eğitim AYNI save_ckpt/load_ckpt
fonksiyonunu kullanır. İki ayrı save yolu tutmak, config alanları ayrışınca
sessizce resume'u kırar ve iyi checkpoint'in üzerine yazdırır.

Yerelin işi: sanity + debug. Tam eğitim uzak sunucuda.

Kullanım:
  python train_local.py --smoke                       # 3 dk, boru hattı testi
  python train_local.py --data ~/ai-data/vexira/tokenized/vexira --preset tiny
  ./afk_train.sh --max-hours 9                        # gece AFK
"""

import os as _os, sys as _sys
# Bu dosya training/ altında; config.py, model.py, tokenizer.py KÖKTE.
# Kök dizin yola eklenmezse import patlar.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import os
import shlex
import sys

POOL = os.environ.get("VEXIRA_POOL", os.path.expanduser("~/ai-data/vexira"))
DEFAULT_DATA = os.path.join(POOL, "tokenized", "vexira")


def main():
    argv = sys.argv[1:]

    smoke = "--smoke" in argv
    if smoke:
        argv.remove("--smoke")

    def has(flag):
        return any(a == flag or a.startswith(flag + "=") for a in argv)

    # Yerel varsayılanlar — uzak eğitimden farklı OLMASI GEREKENLER
    defaults = {
        "--data": DEFAULT_DATA,
        "--out": "models/vexira_local.pt",   # uzak eğitim ckpt'ini EZMESİN
        "--device": "cpu",
        "--compile": "0",                    # CPU'da compile kazancı belirsiz, derleme pahalı
        "--max-hours": "9",                  # gece AFK
        "--eval-every": "500",
        "--log-every": "25",
    }
    if smoke:
        defaults.update({"--preset": "tiny", "--max-steps": "60",
                         "--eval-every": "30", "--log-every": "10",
                         "--epochs": "1", "--max-hours": "0.2"})

    for k, v in defaults.items():
        if not has(k):
            argv += [k, v]

    # CPU'da thread sayısı: E-core'lar GEMM'de zarar verebilir, P-core sayısı
    # genelde daha iyi. bench.py ile ölç, OMP_NUM_THREADS ile geçersiz kıl.
    if "--device" in argv and argv[argv.index("--device") + 1] == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "8")

    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "train.py")] + argv
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
