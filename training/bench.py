# -*- coding: utf-8 -*-
"""
Vexira benchmark — tahmin değil ÖLÇÜM.

Eğitim hızını ölçer ve doğrudan "hedef veri kaç gece sürer" sütununa çevirir.

Ayrıca:
  - torch.set_num_threads taraması (E-core'lar zarar verebilir)
  - torch.compile açık/kapalı
  - pos_encoding varyantları
  - çıkarım ms/satır (greedy vs beam)

Kullanım:
  python bench.py                      # varsayılan tarama
  python bench.py --preset main --threads 4 8 12
  python bench.py --device xpu         # ayrı venv'den
"""

import os as _os, sys as _sys
# Bu dosya training/ altında; config.py, model.py, tokenizer.py KÖKTE.
# Kök dizin yola eklenmezse import patlar.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os
import time

import torch

from config import get_config
from dataset import BitextStore, LengthBucketBatcher, collate, make_synthetic
from model import Vexira

NIGHT_HOURS = 10.0
TARGET_PAIRS = 24_000_000


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "xpu":
        torch.xpu.synchronize()


def bench_train(cfg, store, device, threads=None, compile_=False,
                steps=12, warmup=3, max_tokens=4096):
    if threads:
        torch.set_num_threads(threads)
    model = Vexira(cfg).to(device)
    if device.type in ("cuda", "xpu"):
        cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (9, 0)
        model.amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    model.train()
    if compile_:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98))
    scaler = torch.amp.GradScaler(
        device.type, enabled=(device.type == "cuda"
                              and torch.cuda.get_device_capability(device)[0] < 8))

    idx, _ = store.split()
    batches = LengthBucketBatcher(store, idx, max_tokens=max_tokens).batches(0)
    batches = batches[:steps + warmup]
    if len(batches) < steps + warmup:
        batches = (batches * ((steps + warmup) // max(1, len(batches)) + 1))[:steps + warmup]

    n_pairs = n_tok = 0
    t0 = None
    for i, b in enumerate(batches):
        src, ti, to, sp = collate(store, b, device=device)
        loss = model(src, ti, to, src_pad=sp)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss.mean()).backward()
        scaler.step(opt)
        scaler.update()
        if i == warmup - 1:
            _sync(device)
            t0 = time.time()
        elif i >= warmup:
            n_pairs += len(b)
            n_tok += int(sp.sum()) + int((to != -100).sum())
    _sync(device)
    el = time.time() - t0
    return dict(pairs_s=n_pairs / el, tok_s=n_tok / el,
                nights=TARGET_PAIRS / (n_pairs / el) / 3600 / NIGHT_HOURS)


@torch.no_grad()
def bench_infer(cfg, device, threads=None, n=64, src_len=24, beam=1, max_new=32):
    if threads:
        torch.set_num_threads(threads)
    model = Vexira(cfg).to(device).eval()
    src = torch.randint(20, cfg.vocab_size, (n, src_len), device=device)
    src[:, 0] = 4
    src_pad = torch.ones_like(src, dtype=torch.bool)

    for rep in range(2):                       # 1. tur warmup
        _sync(device)
        t0 = time.time()
        mem = model.encode(src, src_pad)
        if beam > 1:
            mem = mem.repeat_interleave(beam, dim=0)
        caches = model.init_cache(mem, max_len=max_new + 1)
        ys = torch.full((mem.size(0), 1), 2, dtype=torch.long, device=device)
        for t in range(max_new):
            logits = model.decode_step(ys[:, -1:], caches, t)[:, -1]
            ys = torch.cat([ys, logits.argmax(-1, keepdim=True)], dim=1)
        _sync(device)
        el = time.time() - t0
    return dict(ms_line=1000 * el / n, lines_s=n / el)


def main():
    ap = argparse.ArgumentParser(description="Vexira benchmark")
    ap.add_argument("--preset", default="main", choices=["tiny", "main"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, nargs="+", default=None)
    ap.add_argument("--data", default=None, help="gerçek .bin prefix (yoksa sentetik)")
    ap.add_argument("--steps", type=int, default=12,
                    help="CPU+main'de otomatik 4'e iner")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--pos-ab", action="store_true", help="pos_encoding varyantlarını karşılaştır")
    args = ap.parse_args()

    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available()
                           else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                                 else "cpu"))
    else:
        dev = torch.device(args.device)

    print(f"cihaz     : {dev}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        print(f"gpu       : {p.name}  {p.total_memory/1e9:.1f} GB  cc{p.major}.{p.minor}")
    print(f"torch     : {torch.__version__}")
    print(f"cpu thread: {torch.get_num_threads()} (max {torch.get_num_interop_threads()} interop)")

    if args.data:
        store = BitextStore(args.data)
        print(f"veri      : {args.data} ({len(store):,} çift)")
    else:
        import tempfile
        d = tempfile.mkdtemp()
        make_synthetic(f"{d}/bench", n=6000, src_len=(8, 40), tgt_len=(8, 40))
        store = BitextStore(f"{d}/bench")
        print("veri      : sentetik (6000 çift, uzunluk 8-40)")

    cfg = get_config(args.preset, vocab_size=store.vocab_size)
    bd = cfg.param_breakdown()
    print(f"config    : {args.preset} {cfg.n_enc_layer}enc+{cfg.n_dec_layer}dec "
          f"d={cfg.d_model} ff={cfg.d_ff} -> {bd['total']/1e6:.1f}M param")

    # CPU'da 80M modelin tek adımı ~50 sn. Varsayılan olarak 3 thread ayarını
    # taramak 35+ dakika sürer ve kimse beklemez -> tek ayar koş, tarama opt-in.
    if args.threads:
        thread_list = args.threads
    elif dev.type == "cpu":
        thread_list = [min(8, os.cpu_count() or 8)]     # P-core sayısı iyi başlangıç
        print(f"\n(CPU: tek thread ayarı [{thread_list[0]}] koşuluyor. "
              f"Tarama için: --threads 4 8 12)")
    else:
        thread_list = [None]

    # CPU'da 'main' pahalı -> adım sayısını kırp, kullanıcıya süreyi söyle
    if dev.type == "cpu" and args.preset == "main" and args.steps > 4:
        print(f"(CPU + main: --steps {args.steps} -> 4'e indirildi. "
              f"Zorlamak için --steps ile açıkça ver.)")
        args.steps = 4

    print(f"\n{'='*78}\nEĞİTİM  (hedef {TARGET_PAIRS/1e6:.0f}M çift, gece {NIGHT_HOURS:.0f}s)")
    print(f"{'thread':>7} {'compile':>8} {'çift/sn':>10} {'token/sn':>11} {'gece':>9}")
    print("-" * 78)
    best = None
    for th in thread_list:
        for comp in ([False, True] if args.compile else [False]):
            try:
                r = bench_train(cfg, store, dev, th, comp, args.steps,
                                max_tokens=args.max_tokens)
            except Exception as e:                              # noqa: BLE001
                print(f"{str(th):>7} {str(comp):>8}  HATA {type(e).__name__}: {str(e)[:40]}")
                continue
            print(f"{str(th):>7} {str(comp):>8} {r['pairs_s']:>10.1f} "
                  f"{r['tok_s']:>11.0f} {r['nights']:>8.0f}")
            if best is None or r["pairs_s"] > best[1]["pairs_s"]:
                best = (th, r)

    print(f"\n{'='*78}\nÇIKARIM  (24 token kaynak -> 32 token hedef, KV cache)")
    print(f"{'thread':>7} {'beam':>5} {'ms/satır':>10} {'satır/sn':>10}")
    print("-" * 78)
    for th in thread_list:
        for beam in (1, 2, 4):
            try:
                r = bench_infer(cfg, dev, th, beam=beam)
            except Exception as e:                              # noqa: BLE001
                print(f"{str(th):>7} {beam:>5}  HATA {type(e).__name__}")
                continue
            print(f"{str(th):>7} {beam:>5} {r['ms_line']:>10.1f} {r['lines_s']:>10.1f}")

    if args.pos_ab:
        print(f"\n{'='*78}\npos_encoding hız karşılaştırması")
        print(f"{'pos':>12} {'çift/sn':>10} {'ms/satır':>10}")
        print("-" * 78)
        for pe in ("learned", "sinusoidal", "rope"):
            c = get_config(args.preset, vocab_size=store.vocab_size, pos_encoding=pe)
            tr = bench_train(c, store, dev, best[0] if best else None,
                             False, args.steps, max_tokens=args.max_tokens)
            inf = bench_infer(c, dev, best[0] if best else None)
            print(f"{pe:>12} {tr['pairs_s']:>10.1f} {inf['ms_line']:>10.1f}")
        print("(hız yakınsa KALİTE belirler -> train.py --pos-encoding ile A/B eğit)")

    if best:
        th, r = best
        print(f"\n{'='*78}")
        print(f"EN İYİ: {r['pairs_s']:.1f} çift/sn (thread={th})")
        print(f"  -> {TARGET_PAIRS/1e6:.0f}M çift = {r['nights']:.0f} gece "
              f"({NIGHT_HOURS:.0f}s/gece)")
        print(f"  -> 12 saatlik tek oturum = {r['pairs_s']*12*3600/1e6:.2f}M çift")
        if dev.type == "cpu" and r["nights"] > 30:
            print("\n  Yerel CPU tam eğitim için yeterli değil (beklenen sonuç).")
            print("  Yerel = sanity/debug. Tam eğitim uzak sunucuda.")


if __name__ == "__main__":
    main()
