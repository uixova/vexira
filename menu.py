# -*- coding: utf-8 -*-
"""
Vexira — etkileşimli menü. Terminal bilmeyen için tek tıkla çeviri.

    ./vexira.sh          Linux / macOS
    vexira.bat           Windows (çift tık)
    python menu.py       her yerde

Model BİR KEZ yüklenir ve menü boyunca bellekte kalır: yükleme ~2 saniye,
her işlemde tekrarlamak menüyü kullanılmaz yapardı.

Çıktı hem terminale basılır hem dosyaya yazılır — "nolur nolmaz" ilkesi:
terminal kapanınca çeviri kaybolmasın.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

C = {"t": "\033[38;5;75m", "g": "\033[38;5;79m", "d": "\033[38;5;245m",
     "y": "\033[38;5;221m", "r": "\033[38;5;203m", "b": "\033[1m", "x": "\033[0m"}
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    C = {k: "" for k in C}          # eski cmd.exe ANSI bilmiyor


def c(k, s):
    return f"{C[k]}{s}{C['x']}"


def ask(prompt, default=None):
    try:
        v = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return v or default


def out_path(src_path, to):
    base, ext = os.path.splitext(src_path)
    return f"{base}.{to}{ext or '.txt'}"


def save_text(lines, to):
    os.makedirs("translate", exist_ok=True)
    p = os.path.join("translate", f"ceviri_{to}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p


def do_text(tr, to):
    """Satır yaz, Enter — ANINDA çevrilir. Boş satır menüye döner.

    Önce "hepsini yaz, boş satırda bitir" tasarımıydı: kullanıcı tek cümle için
    bile İKİ kez Enter'a basmak zorunda kalıyordu ve çeviriyi görene kadar
    bekliyordu. Satır başına anında cevap hem daha az tuş hem daha canlı.
    """
    arrow = "EN → TR" if to == "tr" else "TR → EN"
    print(c("d", f"\n  {arrow} · satır yaz + Enter = çeviri · boş Enter = menü\n"))
    done = []
    while True:
        try:
            ln = input(c("t", "  > "))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not ln.strip():
            break
        t0 = time.time()
        # domain="doc": serbest metin için doğru alan. "sub" altyazı DOSYASI
        # için; orada kısa selamlamalar iki konuşmacılı diyaloğa dönüşüyor
        # ("Good morning." -> "Günaydın. - Günaydın.").
        r = tr.translate([ln], tgt_lang=to, domain="doc", beam=2)[0]
        print(c("g", f"    {r}") + c("d", f"   ({1000*(time.time()-t0):.0f} ms)"))
        done.append(r)
    if done:
        print(c("d", f"\n  {len(done)} satır → {save_text(done, to)}"))


def do_file(tr, to):
    p = ask(c("d", "\n  Dosya yolu: "))
    if not p:
        return
    p = os.path.expanduser(p.strip().strip('"').strip("'"))
    if not os.path.isfile(p):
        print(c("r", f"  dosya yok: {p}"))
        return
    lines = [l.rstrip("\n") for l in open(p, encoding="utf-8", errors="replace")]
    idx = [i for i, l in enumerate(lines)
           if l.strip() and not l.lstrip().startswith("#")]
    if not idx:
        print(c("r", "  çevrilecek satır yok"))
        return
    print(c("d", f"  {len(idx)} satır çevriliyor…"))
    t0 = time.time()
    got = tr.translate([lines[i] for i in idx], tgt_lang=to, domain="doc", beam=2)
    el = time.time() - t0
    # Boş satır ve yorumlar YERİNDE kalsın — çıktı kaynakla hizalı olsun.
    res = list(lines)
    for i, g in zip(idx, got):
        res[i] = g
    dst = out_path(p, to)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(res) + "\n")
    print(c("g", f"\n  ✓ {len(got)} satır · {el:.1f} sn · {1000*el/len(got):.0f} ms/satır"))
    print(c("g", f"  → {dst}"))
    print(c("d", "\n  ilk 5 satır:"))
    for s, g in list(zip([lines[i] for i in idx], got))[:5]:
        print(c("d", f"    {s[:60]}"))
        print(f"    {g[:60]}\n")


def main():
    print(c("t", "\n  ╭──────────────────────────────────────╮"))
    print(c("t", "  │") + c("b", "  Vexira — TR ↔ EN çeviri            ") + c("t", "│"))
    print(c("t", "  ╰──────────────────────────────────────╯"))
    print(c("d", "  model yükleniyor…"), end="", flush=True)

    try:
        from translate import Translator, DEFAULT_CKPT
    except SystemExit as e:                 # eksik bağımlılık mesajı
        print(f"\n{e}")
        ask("\n  Enter…")
        return 1

    if not os.path.exists(DEFAULT_CKPT):
        print(c("r", f"\n\n  Model yok: {DEFAULT_CKPT}\n"))
        print("  İndir:")
        print(c("y", "    huggingface-cli download uixova/vexira "
                     "vexira_sft.pt vexira_spm.model --local-dir models/\n"))
        ask("  Enter…")
        return 1

    t0 = time.time()
    tr = Translator()
    i = tr.info
    print(f"\r  {c('g', 'hazır')} — {i['params']/1e6:.1f}M param · "
          f"sözlük {i['glossary']} terim · {i['runtime']['threads']} thread "
          f"({time.time()-t0:.1f} sn)          ")

    while True:
        print(c("t", "\n  ─────────────────────────────"))
        print("   1  " + c("b", "Metin çevir") + c("d", "   EN → TR"))
        print("   2  " + c("b", "Metin çevir") + c("d", "   TR → EN"))
        print("   3  " + c("b", "Dosya çevir") + c("d", "   EN → TR"))
        print("   4  " + c("b", "Dosya çevir") + c("d", "   TR → EN"))
        print("   5  " + c("b", "Tarayıcı arayüzü"))
        print("   q  " + c("d", "çıkış"))
        s = ask(c("t", "\n  seçim: "))
        if s is None or s.lower() in ("q", "quit", "exit", "0"):
            print(c("d", "\n  görüşürüz.\n"))
            return 0
        if s == "1":
            do_text(tr, "tr")
        elif s == "2":
            do_text(tr, "en")
        elif s == "3":
            do_file(tr, "tr")
        elif s == "4":
            do_file(tr, "en")
        elif s == "5":
            print(c("d", "\n  webui başlatılıyor — durdurmak için Ctrl+C\n"))
            os.system(f'"{sys.executable}" webui.py')
        else:
            print(c("r", "  geçersiz seçim"))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(c("d", "\n  iptal\n"))
