# -*- coding: utf-8 -*-
"""
Vexira değerlendirme — sacrebleu BLEU + chrF++, FLORES-200 devtest.

chrF++ neden BLEU'nun yanında: BLEU kelime n-gram'ı sayar. Türkçe eklemeli —
"evden" ve "eve" BLEU'ya göre tamamen farklı iki kelime, oysa kök aynı ve model
neredeyse doğru. chrF++ karakter n-gram'ı kullandığı için bu kısmi doğruluğu
görür. Eklemeli dillerde insan yargısıyla korelasyonu BLEU'dan yüksek.

Tavan referansı: --compare ile Helsinki-NLP/opus-mt-tc-big-en-tr aynı test
setinde bir kez koşulur (CPU'da ~15 dk). Kendi modelin kaç puan altında net görülür.

Kullanım:
  python evaluate.py --ckpt models/vexira.pt
  python evaluate.py --ckpt models/vexira.pt --quick 200      # eğitim sırasında
  python evaluate.py --ckpt models/vexira.pt --ctx-ab         # <ctx> faydası
  python evaluate.py --compare                                # tavan referansı
"""

import argparse
import json
import os
import sys
import time

POOL = os.environ.get("VEXIRA_POOL", os.path.expanduser("~/ai-data/vexira"))
DEVTEST = os.path.join(POOL, "eval", "flores200_devtest.jsonl")


def load_pairs(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def score(hyps, refs):
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)   # chrF++
    return {"bleu": round(bleu.score, 2), "chrf2": round(chrf.score, 2),
            "bleu_sig": bleu.format(width=2)}


def eval_direction(tr, rows, tgt_lang, beam, ctx_lines=0, batch=32, domain="doc"):
    """tgt_lang='tr' -> en2tr,  'en' -> tr2en"""
    if tgt_lang == "tr":
        srcs = [r["src"] for r in rows]
        refs = [r["tgt"] for r in rows]
    else:
        srcs = [r["tgt"] for r in rows]
        refs = [r["src"] for r in rows]

    ctx = None
    if ctx_lines:
        # FLORES satırları aynı belgeden ardışık değil; bağlam testi için
        # önceki satırı yine de veriyoruz — gerçek kazanç altyazıda görülecek.
        ctx = [srcs[max(0, i - ctx_lines):i] for i in range(len(srcs))]

    t0 = time.time()
    hyps = tr.translate(srcs, tgt_lang=tgt_lang, ctx=ctx, domain=domain,
                        beam=beam, batch_size=batch)
    el = time.time() - t0
    s = score(hyps, refs)
    s["ms_line"] = round(1000 * el / max(1, len(srcs)), 1)
    s["n"] = len(srcs)
    return s, hyps, refs


def compare_baseline(rows, limit=0):
    """Tavan referansı: opus-mt-tc-big-en-tr (~230M). Tek seferlik."""
    try:
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError:
        print("transformers yok:  pip install --user --break-system-packages transformers")
        return None
    import torch
    name = "Helsinki-NLP/opus-mt-tc-big-en-tr"
    print(f"[tavan] {name} yükleniyor (ilk seferde indirir)...")
    tok = MarianTokenizer.from_pretrained(name)
    m = MarianMTModel.from_pretrained(name).eval()
    srcs = [r["src"] for r in rows]
    refs = [r["tgt"] for r in rows]
    hyps = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(srcs), 16):
            b = tok(srcs[i:i + 16], return_tensors="pt", padding=True, truncation=True)
            out = m.generate(**b, num_beams=4, max_new_tokens=256)
            hyps += tok.batch_decode(out, skip_special_tokens=True)
            if i % 160 == 0:
                print(f"  {i}/{len(srcs)}  {time.time()-t0:.0f}s", flush=True)
    s = score(hyps, refs)
    s["ms_line"] = round(1000 * (time.time() - t0) / len(srcs), 1)
    s["n"] = len(srcs)
    return s


def consistency(tr_engine, lines, tgt_lang, domain, beam):
    """TERİM TUTARLILIĞI ölçümü — BLEU'nun göremediği kusur.

    BLEU cümle başına bakar; "renderer"ı bir satırda "oluşturucu", diğerinde
    "renderer" çevirmek BLEU'yu neredeyse hiç düşürmez ama kullanıcı için
    dosyayı bozar. Burada ölçülen: aynı kaynak dizesi kaç FARKLI karşılık aldı.

    Not: tekrar eden dizeler zaten aynı çıkar (model deterministik). Asıl sinyal
    aynı TERİMİN farklı cümlelerde farklı çevrilmesi — o yüzden kelime düzeyinde
    de sayılıyor.
    """
    from collections import Counter, defaultdict
    import re as _re
    outs = tr_engine.translate(lines, tgt_lang, domain=domain, beam=beam)

    # 1) birebir aynı kaynak satırı -> farklı çıktı
    by_line = defaultdict(set)
    for s, o in zip(lines, outs):
        by_line[s.strip()].add(o.strip())
    line_bad = {k: v for k, v in by_line.items() if len(v) > 1}

    # 2) kısa terim (1-3 kelime) tek başına da geçiyorsa, uzun cümledeki
    #    karşılığıyla tutarlı mı
    short = {s.strip().lower(): o.strip() for s, o in zip(lines, outs)
             if 1 <= len(s.split()) <= 3}
    term_bad = defaultdict(set)
    for s, o in zip(lines, outs):
        if len(s.split()) <= 3:
            continue
        low = o.lower()
        for term, canon in short.items():
            if _re.search(r"(?<!\w)" + _re.escape(term) + r"(?!\w)", s.lower()):
                if canon and canon.lower() not in low:
                    term_bad[term].add(o.strip())
    return outs, line_bad, term_bad


def main():
    ap = argparse.ArgumentParser(description="Vexira değerlendirme")
    ap.add_argument("--ckpt", default="models/vexira.pt")
    ap.add_argument("--spm", default=os.path.join(POOL, "spm", "vexira_spm.model"))
    ap.add_argument("--test", default=DEVTEST)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--quick", type=int, default=0, help="ilk N cümle (eğitim sırasında)")
    ap.add_argument("--dirs", nargs="+", default=["tr", "en"],
                    help="hedef diller: tr (en->tr), en (tr->en)")
    ap.add_argument("--ctx-ab", action="store_true", help="<ctx> var/yok karşılaştırması")
    ap.add_argument("--compare", action="store_true", help="opus-mt tavan referansı")
    ap.add_argument("--dump", help="çevirileri bu dosyaya yaz (elle okumak için)")
    ap.add_argument("--glossary", action="store_true",
                    help="sözlüğü AÇ (varsayılan kapalı). FLORES düz metindir; "
                         "sözlük açıkken ölçüm modelin kendi çevirisini değil "
                         "sözlüğün isabetini ölçer, oturumlar kıyaslanamaz olur")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--consistency",
                    help="satır başına bir kaynak dize içeren dosya — terim "
                         "tutarlılığı ölçümü (BLEU bunu göremez)")
    ap.add_argument("--domain", default="ui", choices=["sub", "ocr", "ui", "doc"])
    args = ap.parse_args()

    if args.consistency:
        import torch
        from translate import Translator
        if args.threads:
            torch.set_num_threads(args.threads)
        lines = [l.rstrip("\n") for l in open(args.consistency, encoding="utf-8")
                 if l.strip()]
        tgt = args.dirs[0] if args.dirs else "tr"
        print(f"{len(lines)} dize · domain <{args.domain}> · hedef {tgt}")
        for use_g, name in ((False, "SÖZLÜKSÜZ"), (True, "SÖZLÜKLÜ")):
            eng = Translator(args.ckpt, args.spm, args.device, use_glossary=use_g)
            if use_g and not len(eng.gloss):
                print(f"\n[{name}] ckpt'de sözlük yok — atlandı "
                      f"(python glossary.py build ...)")
                continue
            outs, lb, tb = consistency(eng, lines, tgt, args.domain, args.beam)
            print(f"\n[{name}] {len(eng.gloss)} terim · "
                  f"{len(lb)} çelişkili dize · {len(tb)} tutarsız terim")
            for t, alts in list(tb.items())[:12]:
                print(f"    {t!r}: cümle içinde farklı -> {list(alts)[:2]}")
            if args.dump:
                with open(f"{args.dump}.{name.lower()}", "w", encoding="utf-8") as f:
                    for s, o in zip(lines, outs):
                        f.write(f"{s}\t{o}\n")
        return 0

    if not os.path.exists(args.test):
        print(f"test seti yok: {args.test}\nönce: python data/fetch_bitext.py --flores")
        return 1
    rows = load_pairs(args.test, args.quick)
    print(f"test: {args.test}  ({len(rows)} cümle)")

    if args.compare:
        s = compare_baseline(rows)
        if s:
            print(f"\n[TAVAN] opus-mt-tc-big-en-tr  en->tr  "
                  f"BLEU {s['bleu']}  chrF++ {s['chrf2']}  {s['ms_line']} ms/satır")
        return 0

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
    from translate import Translator
    # FLORES düz metin; sözlük açık ölçüm, modelin öğrendiğini değil sözlüğün
    # isabetini ölçer. Karşılaştırılabilirlik için varsayılan KAPALI.
    tr = Translator(args.ckpt, args.spm, args.device,
                    use_glossary=args.glossary)
    print(f"model: {tr.info['params']/1e6:.1f}M param, adım {tr.info['step']}, "
          f"val {tr.info['best_val']}")

    results = {}
    dumps = {}
    print(f"\n{'yön':<8} {'BLEU':>7} {'chrF++':>8} {'ms/satır':>10}")
    print("-" * 38)
    for tgt in args.dirs:
        name = "en->tr" if tgt == "tr" else "tr->en"
        s, hyps, refs = eval_direction(tr, rows, tgt, args.beam)
        results[name] = s
        dumps[name] = (hyps, refs)
        print(f"{name:<8} {s['bleu']:>7.2f} {s['chrf2']:>8.2f} {s['ms_line']:>10.1f}")

    if args.ctx_ab:
        print(f"\n{'='*38}\n<ctx> BAĞLAM A/B")
        print(f"{'yön':<8} {'ctx yok':>9} {'ctx var':>9} {'fark':>7}")
        print("-" * 38)
        for tgt in args.dirs:
            name = "en->tr" if tgt == "tr" else "tr->en"
            s0 = results[name]
            s1, _, _ = eval_direction(tr, rows, tgt, args.beam, ctx_lines=2)
            d = s1["bleu"] - s0["bleu"]
            print(f"{name:<8} {s0['bleu']:>9.2f} {s1['bleu']:>9.2f} {d:>+7.2f}")
        print("(fark negatifse build_bin.py --ctx-prob 0 ile bağlamı kapat)")

    print(f"\n{'='*38}\nHEDEF: en->tr BLEU >= 22, tr->en >= 25")
    for name, s in results.items():
        want = 22 if name == "en->tr" else 25
        mark = "✓" if s["bleu"] >= want else "→"
        print(f"  {mark} {name}: {s['bleu']:.2f} / {want}")

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for name, (hyps, refs) in dumps.items():
                srcs = ([r["src"] for r in rows] if name == "en->tr"
                        else [r["tgt"] for r in rows])
                f.write(f"\n{'='*70}\n{name}\n{'='*70}\n")
                for s_, h, r_ in zip(srcs, hyps, refs):
                    f.write(f"KAYNAK: {s_}\nÇEVİRİ: {h}\nREFERANS: {r_}\n\n")
        print(f"\nçeviriler -> {args.dump}  (elle oku, ek/ünlü uyumu hatalarına bak)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
