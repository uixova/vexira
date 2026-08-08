# -*- coding: utf-8 -*-
"""
Vexira'yı KENDİ projene bağlama — 3 satır.

    python examples/integrate.py

Senaryo: elinde başka bir model/script var (LLM, OCR, altyazı aracı, oyun
yerelleştirme boru hattı) ve çeviriyi Vexira'ya yaptıracaksın. `translate.py`
dosyasının 650 satırını OKUMAN GEREKMİYOR — ondan tek şey alıyorsun:

    from translate import Translator
    tr = Translator()
    out = tr.translate(["Save"], tgt_lang="tr", domain="ui")

Bu üç satır her şeyi getirir: sözlük, yer tutucu onarımı, tekrar
sadeleştirme, 128 token üstü bölme, toplu iş, doğru thread sayısı.
`Translator` ENTEGRASYON ARAYÜZÜDÜR; minimal.py ise "içeride ne oluyor"u
gösteren öğretici sürüm — üretimde onu kullanma, onarımlar orada yok.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from translate import Translator                                  # noqa: E402


def demo_basic():
    print("── 1. Temel: tek nesne, çok çağrı ──")
    tr = Translator()                     # modeli BİR KEZ yükle, sakla
    print(" ", tr.translate(["Hello world"], tgt_lang="tr")[0])
    print(" ", tr.translate(["Merhaba dünya"], tgt_lang="en")[0])
    return tr


def demo_batch(tr):
    """Toplu iş 2.2x hızlı: 139 ms/satır -> 62 ms/satır.
    Satırları TEK ÇAĞRIDA ver; döngü içinde tek tek çağırma."""
    print("\n── 2. Toplu iş — döngüde tek tek çağırma ──")
    lines = ["Save", "Cancel", "Are you sure you want to quit?",
             "The file could not be opened."]
    for s, o in zip(lines, tr.translate(lines, tgt_lang="tr", domain="ui")):
        print(f"  {s:38s} -> {o}")


def demo_domain(tr):
    """domain çıktıyı gerçekten değiştirir — yanlış seçim kaliteyi düşürür."""
    print("\n── 3. domain seçimi ──")
    s = "The renderer failed to start."
    for d in ("ui", "doc", "sub"):
        print(f"  {d:4s} -> {tr.translate([s], tgt_lang='tr', domain=d)[0]}")
    print("  ui: sözlük devrede · doc: düzyazı · sub: altyazı satırı")


def demo_stats(tr):
    """Kaç satır modele gitti, kaçı sözlükten geldi — maliyet takibi."""
    print("\n── 4. İstatistik (kendi log'una yaz) ──")
    before = dict(tr.stats)
    tr.translate(["Save", "Cancel", "A sentence the glossary does not know."],
                 tgt_lang="tr", domain="ui")
    d = {k: tr.stats[k] - before[k] for k in tr.stats}
    print(f"  sözlükten (model çağrılmadı): {d['exact']}")
    print(f"  modelden                    : {d['model']}")
    print(f"  uzun olduğu için bölünen    : {d['split']}")


def demo_glossary(tr):
    """Kendi terimlerini ekle — projenin sözlüğü modelinkini ezer."""
    print("\n── 5. Kendi terim sözlüğün ──")
    from glossary import Glossary
    mine = Glossary([{"en": "checkpoint", "tr": "kontrol noktası", "domain": "ui"},
                     {"en": "shader", "tr": "gölgelendirici", "domain": "ui"}])
    tr2 = Translator(glossary=mine)       # TSV yolu da verilebilir
    for s in ("checkpoint", "shader"):
        print(f"  {s:12s} -> {tr2.translate([s], tgt_lang='tr', domain='ui')[0]}")
    print("  (Translator(glossary='terimlerim.tsv') de olur)")


def demo_rules(tr):
    """Onarım kuralları modelin İÇİNDE — başka dilde de okunabilir."""
    print("\n── 6. Kurallar modelden geliyor ──")
    for name, r in tr.pp.items():
        why = r.get("why", "")
        print(f"  {name:15s} {r['pattern'][:34]:36s} {why[:44]}")
    print("  ck['postprocess'] — JSON'a yazılıp Rust/JS/C++ tarafından okunabilir")


def demo_server():
    print("\n── 7. Python DIŞINDAN kullanım ──")
    print("  python translate.py --server      # stdin/stdout JSON")
    print('  {"lines":["Hello"],"to":"tr","domain":"ui"}')
    print('  -> {"ok":true,"out":["Merhaba"],"ms":23.1}')
    print("  Node/Go/Rust tarafından alt süreç olarak çağrılabilir;")
    print("  model bir kez yüklenir, istekler boyunca bellekte kalır.")


if __name__ == "__main__":
    tr = demo_basic()
    demo_batch(tr)
    demo_domain(tr)
    demo_stats(tr)
    demo_glossary(tr)
    demo_rules(tr)
    demo_server()
    print()
