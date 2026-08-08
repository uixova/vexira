# -*- coding: utf-8 -*-
"""
Vexira eğitim döngüsü — çok GPU'lu uzak sunucu hedefli, yerelde CPU'da da koşar.

Bellek/hız optimizasyonları:
   1  DDP + torchrun, non-final micro-batch'te no_sync()
   2  nn.DataParallel fallback
   3  bitsandbytes AdamW8bit + otomatik kurulum + fp32 AdamW fallback
   4  seçici gradient checkpointing (ilk N blok)
   5  torch.compile — DDP sarmadan ÖNCE ham modele
   6  chunked cross-entropy (model.py içinde)
   7  autocast model.forward'ın İÇİNDE (DataParallel thread-locality)
   8  eğitimde forward yalnız skaler loss döndürür (DP dengesi)
   9  VRAM'e göre otomatik batch/accum/ckpt ayarı
  10  süre limitli duruş — rank0 karar verir + broadcast (oturum kesilmeden kaydet)
  11  resume: config uyuşmazsa çökme değil sıfırdan başla; optim state ayrı try
  12  fp16'da GradScaler, bf16'da kapalı
  13  memmap uint16 veri (dataset.py)
  14  veri küçükse tamamını RAM'e al

Çeviriye özel: AdamW betas (0.9, 0.98), inverse-sqrt LR, token bütçeli
uzunluk-kovalı batch, label smoothing 0.1.

Kullanım:
  python train.py --data ~/ai-data/vexira/tokenized/vexira --preset main
  torchrun --nproc_per_node=2 train.py --data ... --preset main   # 2 GPU
"""

import os as _os, sys as _sys
# Bu dosya training/ altında; config.py, model.py, tokenizer.py KÖKTE.
# Kök dizin yola eklenmezse import patlar.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from config import get_config
from dataset import (BitextStore, LengthBucketBatcher, collate, prefetch,
                     shard_batches)
from model import Vexira

# ------------------------------ ORTAM ------------------------------

torch._dynamo.config.suppress_errors = True     # compile patlarsa eager'a düş

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD = int(os.environ.get("WORLD_SIZE", 1))
IS_MAIN = RANK == 0


def log(msg):
    if IS_MAIN:
        print(msg, flush=True)


def pick_device(arg):
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{LOCAL_RANK}" if WORLD > 1 else "cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def auto_tune(device, args):
    """[9] VRAM'e göre batch/accum/checkpointing seç. Token bütçesi cinsinden
    (NMT'de batch örnek sayısıyla değil token sayısıyla ölçülür)."""
    if device.type != "cuda":
        # CPU: bellek bol, hesap kıt. Küçük batch = düşük GEMM verimi, o yüzden
        # yine de makul bir bütçe tut.
        return dict(max_tokens=4096, accum=1, ckpt_enc=0, ckpt_dec=0)
    gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if gb >= 40:                    # A100
        return dict(max_tokens=32768, accum=1, ckpt_enc=0, ckpt_dec=0)
    if gb >= 22:                    # 3090 / A10
        return dict(max_tokens=20480, accum=1, ckpt_enc=0, ckpt_dec=0)
    if gb >= 14:                    # 15-16 GB sınıfı — uzak eğitim hedefi
        return dict(max_tokens=12288, accum=2, ckpt_enc=0, ckpt_dec=0)
    return dict(max_tokens=6144, accum=4, ckpt_enc=6, ckpt_dec=3)


def make_optimizer(model, lr, wd):
    """[3] 8-bit AdamW tercih; yoksa kur; olmazsa fused/normal AdamW.
    Norm ve embedding ağırlıklarına weight decay UYGULANMAZ (NMT'de standart)."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    betas = (0.9, 0.98)          # NMT standardı (LM'lerdeki 0.95 değil)

    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
        except ImportError:
            if IS_MAIN:
                log("[optim] bitsandbytes yok, kuruluyor...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                "bitsandbytes"], check=False)
            if WORLD > 1:
                dist.barrier()
            try:
                import bitsandbytes as bnb
            except ImportError:
                bnb = None
        if bnb is not None:
            log("[optim] AdamW8bit")
            return bnb.optim.AdamW8bit(groups, lr=lr, betas=betas)

    log("[optim] torch AdamW")
    return torch.optim.AdamW(groups, lr=lr, betas=betas,
                             fused=torch.cuda.is_available())


def lr_at(step, warmup, peak_lr, total=0):
    """Ön-eğitim: inverse-sqrt (Vaswani/Marian). Cosine yerine bu, çünkü NMT'de
    toplam adım sayısı veriye göre değişir ve cosine sabit bir ufuk ister;
    inverse-sqrt ufuk bilmeden çalışır, resume'da bozulmaz.

    İnce ayar (total>0): warmup + lineer sönme. Burada toplam adım ÖNCEDEN
    bellidir (küçük set) ve sıfıra inen LR, öğrenilen terimleri son adımlarda
    dalgalandırmadan sabitler."""
    step = max(1, step)
    if step < warmup:
        return peak_lr * step / warmup
    if total:
        frac = (step - warmup) / max(1, total - warmup)
        return peak_lr * max(0.02, 1.0 - frac)
    return peak_lr * math.sqrt(warmup / step)


# ------------------------------ CHECKPOINT ------------------------------

# Ağırlık/optimizer dışında checkpoint'te taşınan, eğitimin ÜRETMEDİĞİ ama
# kaybedilmemesi gereken alanlar (sözlük, gömülü tokenizer). Kaydetmede
# korunmazlarsa ilk commit'te sessizce silinirler.
CARRY_KEYS = ("glossary", "spm_model", "spm_sha")


def save_ckpt(path, model, opt, cfg, step, tokens_seen, pairs_seen, best_val,
              commits, extra=None, carry=None):
    """[11] TEK kaydetme yolu — yerel ve uzak eğitim AYNI fonksiyonu çağırır.

    İki ayrı save yolu tutmak, ikisinin config alanları ayrışınca sessizce
    resume'u kırar (ve --force ile iyi checkpoint'in üzerine yazdırır)."""
    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    raw = getattr(raw, "_orig_mod", raw)          # torch.compile sarmalayıcısı
    tmp = path + ".tmp"
    blob = {"model": raw.state_dict(),
            "optim": opt.state_dict() if opt is not None else None,
            "config": cfg.to_dict(),
            "step": step, "tokens_seen": tokens_seen, "pairs_seen": pairs_seen,
            "best_val": best_val, "commits": commits,
            "extra": extra or {}}
    blob.update(carry or {})
    torch.save(blob, tmp)
    os.replace(tmp, path)                          # atomik — yarım ckpt kalmasın


def read_carry(path):
    """Var olan checkpoint'ten taşınacak alanları oku (yoksa boş)."""
    if not os.path.exists(path):
        return {}
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                            # noqa: BLE001
        return {}
    return {k: ck[k] for k in CARRY_KEYS if k in ck}


def load_ckpt(path, model, opt, cfg):
    """Config uyuşmazsa ÇÖKME, sıfırdan başla. Optim state ayrı try içinde:
    8-bit <-> fp32 optimizer değişimi ağırlıkları kaybettirmesin."""
    if not os.path.exists(path):
        return 0, 0, 0, float("inf"), 0
    ck = torch.load(path, map_location="cpu", weights_only=False)
    old = ck.get("config", {})
    if not cfg.matches(old):
        log(f"[resume] config uyuşmuyor -> SIFIRDAN başlanıyor\n"
            f"         ckpt: {json.dumps(old, ensure_ascii=False)}\n"
            f"         şu an: {json.dumps(cfg.to_dict(), ensure_ascii=False)}")
        return 0, 0, 0, float("inf"), 0

    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    raw = getattr(raw, "_orig_mod", raw)
    raw.load_state_dict(ck["model"])
    if opt is not None and ck.get("optim"):
        try:
            opt.load_state_dict(ck["optim"])
        except Exception as e:                     # noqa: BLE001
            log(f"[resume] optimizer state yüklenemedi ({e}) — ağırlıklar KORUNDU")
    log(f"[resume] adım {ck['step']}, {ck['pairs_seen']/1e6:.2f}M çift, "
        f"best_val {ck['best_val']:.4f}, {ck.get('commits', 0)}. commit")
    return (ck["step"], ck["tokens_seen"], ck["pairs_seen"],
            ck["best_val"], ck.get("commits", 0))


# ------------------------------ EVAL ------------------------------

@torch.no_grad()
def evaluate(model, store, batches, device, max_batches=40):
    model.eval()
    tot, n = 0.0, 0
    for b in batches[:max_batches]:
        src, ti, to, sp = collate(store, b, device=device)
        loss = model(src, ti, to, src_pad=sp)
        tot += float(loss.mean())
        n += 1
    model.train()
    return tot / max(1, n)


# ------------------------------ ANA ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Vexira eğitim")
    ap.add_argument("--data", required=True, help="tokenize prefix (.bin/.idx.npy/.meta.json)")
    ap.add_argument("--out", default="models/vexira.pt")
    ap.add_argument("--preset", default="main", choices=["tiny", "main"])
    ap.add_argument("--pos-encoding", default=None, choices=["learned", "sinusoidal", "rope"])
    ap.add_argument("--n-dec-layer", type=int, default=None, help="Türkçe ek hatası görülürse artır")
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--warmup", type=int, default=4000)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2,
                    help="EZBER RİSKİ: 2 tercih, 3 üst sınır. Hacim epoch tekrarından "
                         "DEĞİL yeni kaynaktan gelmeli (fetch_bitext.py --bulk)")
    ap.add_argument("--chinchilla-mult", type=float, default=3.0,
                    help="hedef token bütçesi = mult x 20 x parametre")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=11.75,
                    help="uzak oturum kesilmeden önce temiz dur + kaydet")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = VRAM'e göre otomatik")
    ap.add_argument("--accum", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--ram-data", default="auto", choices=["auto", "0", "1"],
                    help="[14] bin+idx'i RAM'e al. auto = sığarsa (DDP'de rank "
                         "başına maliyet; zorlamak OOM kill'e yol açar)")
    ap.add_argument("--seed", type=int, default=42)
    # --- ince ayar (SFT) ---
    ap.add_argument("--finetune", action="store_true",
                    help="düşük LR ince ayar: ağırlıkları --init-from'dan al, "
                         "adım/optimizer sıfırla, LR'yi lineer söndür")
    ap.add_argument("--init-from", default=None,
                    help="ince ayarın başlayacağı ön-eğitim checkpoint'i "
                         "(--out'tan FARKLI olmalı: ön-eğitim beyni ezilmesin)")
    args = ap.parse_args()

    # İnce ayarda ön-eğitim varsayılanları yanlış: 7e-4 ile 2 dakikada
    # öğrenilenlerin üstü çizilir (catastrophic forgetting).
    if args.finetune:
        if args.lr == ap.get_default("lr"):
            args.lr = 5e-5
        if args.warmup == ap.get_default("warmup"):
            args.warmup = 200
        if args.epochs == ap.get_default("epochs"):
            args.epochs = 3          # SFT seti küçük; burada 3 epoch ezber değil
        if args.eval_every == ap.get_default("eval_every"):
            args.eval_every = 250

    torch.manual_seed(args.seed + RANK)

    # --- [1] DDP kurulumu ---
    use_ddp = WORLD > 1
    if use_ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(LOCAL_RANK)
    device = pick_device(args.device)
    log(f"[env] device={device} world={WORLD} ddp={use_ddp}")

    # --- veri ---
    ram_mode = args.ram_data if args.ram_data == "auto" else bool(int(args.ram_data))
    store = BitextStore(args.data, ram=ram_mode, world_size=WORLD, verbose=IS_MAIN)
    # Ön-eğitimde %0.2 val zaten 20k örnek. SFT seti binlerle ölçülüyor; aynı
    # oran 60 örnek verir ve val gürültüden ibaret olur.
    tr_idx, va_idx = store.split(val_frac=0.02 if args.finetune else 0.002)
    log(f"[veri] {len(store):,} çift ({store.meta['n_tokens']/1e6:.1f}M token), "
        f"train {len(tr_idx):,} / val {len(va_idx):,}, vocab {store.vocab_size}")

    # --- config ---
    kw = {"vocab_size": store.vocab_size}
    if args.pos_encoding:
        kw["pos_encoding"] = args.pos_encoding
    if args.n_dec_layer:
        kw["n_dec_layer"] = args.n_dec_layer
    cfg = get_config(args.preset, **kw)

    tune = auto_tune(device, args)
    max_tokens = args.max_tokens or tune["max_tokens"]
    accum = args.accum or tune["accum"]

    # --- [12] precision: eski GPU'lar (cc<8) bf16 desteklemez -> fp16 + GradScaler ---
    use_amp = device.type in ("cuda", "xpu")
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16
    elif device.type == "xpu":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None            # CPU: Raptor Lake'te bf16 hızlandırma yok, fp32
    scaler = torch.amp.GradScaler(device.type,
                                  enabled=(use_amp and amp_dtype == torch.float16))

    # --- model ---
    model = Vexira(cfg).to(device)
    model.amp_dtype = amp_dtype if use_amp else None        # [7] autocast forward içinde
    model.grad_ckpt_enc = tune["ckpt_enc"]                  # [4]
    model.grad_ckpt_dec = tune["ckpt_dec"]
    log(f"[model] {cfg.n_enc_layer}enc+{cfg.n_dec_layer}dec d={cfg.d_model} "
        f"ff={cfg.d_ff} pos={cfg.pos_encoding} -> {model.num_params()/1e6:.2f}M param "
        f"({model.num_params(non_embedding=True)/1e6:.2f}M embedding hariç)")
    log(f"[batch] max_tokens={max_tokens} accum={accum} amp={amp_dtype} "
        f"ckpt=({tune['ckpt_enc']},{tune['ckpt_dec']})")

    # --- token bütçesi: bilinçli overtrain hedefi ---
    # Chinchilla optimumu ~20 token/parametre. Küçük modelde bunun katlarına
    # çıkmak kaliteyi artırır ve ÇIKARIM maliyetini hiç değiştirmez — bu boyutta
    # en ucuz kazanç. Hedef mult x 20 x param.
    n_param = model.num_params()
    chin = 20.0 * n_param
    target_tok = args.chinchilla_mult * chin
    tok_per_epoch = store.meta["n_tokens"]
    reach = args.epochs * tok_per_epoch
    if args.finetune:
        # Chinchilla ön-eğitim bütçesi kuralıdır; küçük SFT setinde anlamsız ve
        # uyarıları yanıltıcı olur.
        log(f"[sft] {tok_per_epoch/1e6:.2f}M token/epoch x {args.epochs} epoch "
            f"— bütçe kontrolü kapalı (ön-eğitim değil)")
    else:
        log(f"[bütçe] Chinchilla(20x{n_param/1e6:.1f}M) = {chin/1e9:.2f}B token · "
            f"hedef {args.chinchilla_mult:.1f}x = {target_tok/1e9:.2f}B")
        log(f"        veri {tok_per_epoch/1e9:.2f}B tok/epoch x {args.epochs} epoch "
            f"= {reach/1e9:.2f}B ({reach/chin:.2f}x Chinchilla)")
        # Hedefe epoch ARTIRARAK ulaşmak ezber demek. Eksikse çözüm yeni veri.
        if args.epochs > 3:
            log(f"        ⚠️  --epochs {args.epochs} > 3: EZBER RİSKİ. Bunun yerine "
                f"veri artır: fetch_bitext.py --bulk")
        if reach < target_tok * 0.8:
            need_data = target_tok / max(1, args.epochs)
            log(f"        ⚠️  hedefin altında. Epoch ARTIRMA — {need_data/1e9:.2f}B "
                f"tok/epoch gerekli, yani ~{need_data/max(1,tok_per_epoch):.1f}x daha "
                f"fazla TAZE veri (fetch_bitext.py --bulk / --all)")

    opt = make_optimizer(model, args.lr, args.wd)

    stage = "finetune" if args.finetune else "pretrain"
    # Sözlük/gömülü tokenizer eğitimin ürünü değil ama checkpoint'te yaşıyor.
    # Kaynak: ince ayarda ön-eğitim beyni, aksi hâlde çıktı dosyasının kendisi.
    carry = read_carry(args.init_from) if args.init_from else {}
    carry = carry or read_carry(args.out)

    # --- ince ayar: ağırlığı ön-eğitimden al, sayaçları SIFIRLA ---
    if args.finetune:
        if not args.init_from:
            log("[hata] --finetune için --init-from gerekli (ön-eğitim ckpt'si)")
            return 2
        if os.path.abspath(args.init_from) == os.path.abspath(args.out):
            log(f"[hata] --init-from ve --out aynı dosya: {args.out}\n"
                f"       ince ayar ön-eğitim beynini EZERDİ. Farklı --out ver.")
            return 2
        if os.path.exists(args.out):
            # ince ayar oturumu kesildiyse kendi çıktısından devam et
            step, tokens_seen, pairs_seen, best_val, commits = load_ckpt(
                args.out, model, opt, cfg)
            log(f"[sft] kendi çıktısından devam: adım {step}")
        else:
            # optimizer state ALINMAZ: ön-eğitimin Adam momentleri 7e-4 ölçeğinde,
            # 5e-5 ile taşınırsa ilk adımlarda ağırlıkları savurur.
            src_step, _, _, _, _ = load_ckpt(args.init_from, model, None, cfg)
            if src_step == 0:
                # load_ckpt config uyuşmazlığında sessizce sıfırdan başlatır;
                # ince ayarda bu "rastgele ağırlıkla SFT" demektir, kabul edilemez.
                log(f"[hata] {args.init_from} yüklenemedi ya da config uyuşmuyor "
                    f"— ince ayar rastgele ağırlıkla başlayamaz")
                return 2
            step = tokens_seen = pairs_seen = commits = 0
            best_val = float("inf")
            log(f"[sft] ağırlık {args.init_from} (adım {src_step}) -> sayaç ve "
                f"optimizer SIFIRLANDI, lr {args.lr:.1e} warmup {args.warmup}")
    else:
        step, tokens_seen, pairs_seen, best_val, commits = load_ckpt(
            args.out, model, opt, cfg)

    # --- resume: VERİ KONUMUNU da geri al ---
    # Sadece step'i geri yüklemek yetmez: epoch döngüsü baştan başlar ve daha
    # önce görülen veri TEKRAR işlenir. Bu hem oturum süresini yakar hem aynı
    # örnekleri ikinci kez göstererek ezber riski yaratır.
    # Batch listesi (seed, epoch, max_tokens) ile deterministik, o yüzden
    # konumu saklayıp atlamak birebir doğru.
    resume_epoch = resume_bi = 0
    if step > 0:
        ex = {}
        try:
            ex = torch.load(args.out, map_location="cpu",
                            weights_only=False).get("extra", {}) or {}
        except Exception:                                    # noqa: BLE001
            pass
        if ex.get("max_tokens") == max_tokens and ex.get("world") == WORLD:
            resume_epoch = int(ex.get("epoch", 0))
            resume_bi = int(ex.get("batch_pos", 0))
            log(f"[resume] veri konumu: epoch {resume_epoch}, batch {resume_bi:,} "
                f"-> bu kadarı atlanacak")
        elif ex:
            log(f"[resume] ⚠️  batch düzeni değişmiş "
                f"(max_tokens {ex.get('max_tokens')}->{max_tokens}, "
                f"world {ex.get('world')}->{WORLD}) — veri baştan taranacak")

    # --- [5] compile ÖNCE, DDP sarmalama SONRA ---
    if args.compile and hasattr(torch, "compile") and device.type != "cpu":
        log("[compile] torch.compile açık")
        model = torch.compile(model)
    if use_ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK] if device.type == "cuda" else None)
    elif device.type == "cuda" and torch.cuda.device_count() > 1:
        log(f"[dp] DataParallel {torch.cuda.device_count()} GPU")     # [2]
        model = nn.DataParallel(model)
    model.train()

    train_b = LengthBucketBatcher(store, tr_idx, max_tokens=max_tokens, seed=args.seed)
    val_b = LengthBucketBatcher(store, va_idx, max_tokens=max_tokens,
                                shuffle=False).batches(0)

    t0 = time.time()
    stop = False
    done = False                  # True = veri bitti, süre dolmadan temiz bitiş
    epoch = resume_epoch          # döngü hiç dönmezse de tanımlı olsun
    bi = resume_bi
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    per_rank = len(train_b) // max(1, WORLD)
    # İnce ayarda LR lineer söner. Toplam adım KOŞUNUN TAMAMI olmalı, kalanı
    # değil: `step` resume'da devam ediyor, kalan üzerinden hesaplarsak eğri
    # ikinci oturumda baştan başlar ve LR yanlış yerden düşer.
    total_steps = (args.epochs * per_rank) // max(1, accum) if args.finetune else 0
    log(f"[başla] adım {step}, hedef {args.epochs} epoch, "
        f"~{per_rank} batch/epoch/rank"
        + (f", ~{total_steps} adım (LR lineer söner)" if total_steps else ""))

    for epoch in range(args.epochs):
        if stop:
            break
        if epoch < resume_epoch:                 # tamamlanmış epoch'u atla
            continue
        batches = shard_batches(train_b.batches(epoch), RANK, WORLD)
        start_bi = resume_bi if epoch == resume_epoch else 0
        if start_bi:
            log(f"[resume] epoch {epoch}: {start_bi:,}/{len(batches):,} batch atlanıyor")
        for bi in range(start_bi, len(batches) - accum + 1, accum):
            micro = batches[bi:bi + accum]
            lr = lr_at(step + 1, args.warmup, args.lr, total=total_steps)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            loss_acc, tok_acc, pair_acc = 0.0, 0, 0
            # collate arka planda: memmap okuması GPU hesabıyla örtüşsün
            # (ölçüldü: 9.5 -> 4.6 ms/batch)
            for k, (src, ti, to, sp) in enumerate(
                    prefetch(store, micro, device=device,
                             pin=(device.type == "cuda"))):
                # [1] son micro-batch dışında gradyan senkronizasyonunu atla
                sync_ctx = (model.no_sync() if (use_ddp and k < len(micro) - 1)
                            else _Null())
                with sync_ctx:
                    loss = model(src, ti, to, src_pad=sp)
                    loss = loss.mean() / len(micro)          # [8] skaler + DP dengesi
                    scaler.scale(loss).backward()
                loss_acc += float(loss.detach()) * len(micro)
                tok_acc += int(sp.sum()) + int((to != -100).sum())
                pair_acc += int(sp.size(0))   # mb yok: batch boyutu tensörden

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], args.clip)
            scaler.step(opt)
            scaler.update()

            step += 1
            tokens_seen += tok_acc * WORLD
            pairs_seen += pair_acc * WORLD

            if step % args.log_every == 0:
                el = time.time() - t0
                log(f"  adım {step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | "
                    f"gnorm {float(gnorm):.2f} | {tokens_seen/1e9:.3f}B token "
                    f"({tokens_seen/chin:.2f}x chin) | "
                    f"{pairs_seen/max(1,el):.0f} çift/sn | {el/3600:.2f}h")

            # --- [10] süre limiti: rank0 karar verir, herkese yayar ---
            if step % args.log_every == 0 or step % args.eval_every == 0:
                over = (time.time() - t0) / 3600 >= args.max_hours
                if args.max_steps and step >= args.max_steps:
                    over = True
                if use_ddp:
                    flag = torch.tensor([1 if over else 0], device=device)
                    dist.broadcast(flag, src=0)
                    over = bool(flag.item())
                if over:
                    stop = True

            if step % args.eval_every == 0 or stop:
                vl = evaluate(model, store, val_b, device)
                is_best = vl < best_val
                best_val = min(best_val, vl)
                log(f"[EVAL] adım {step} | val {vl:.4f} | best {best_val:.4f}"
                    + ("  *yeni en iyi*" if is_best else ""))
                if IS_MAIN:
                    commits += 1
                    save_ckpt(args.out, model, opt, cfg, step, tokens_seen,
                              pairs_seen, best_val, commits, carry=carry,
                              extra={"epoch": epoch, "batch_pos": bi,
                                     "max_tokens": max_tokens, "world": WORLD,
                                     "val": vl, "hours": (time.time() - t0) / 3600,
                                     "stage": stage})
                    log(f"[ckpt] {args.out} kaydedildi ({commits}. commit)")
                if use_ddp:
                    dist.barrier()
            if stop:
                break
        else:
            # for döngüsü kırılmadan bitti = bu epoch'un TÜM verisi tükendi.
            # Son epoch'sa oturum burada kapanır; uzak sunucuda script bitince
            # commit hemen tamamlanır, kalan süre boşa yanmaz.
            if epoch == args.epochs - 1:
                done = True
                log(f"[veri] epoch {epoch} tamamlandı — hedef epoch sayısına "
                    f"ulaşıldı, süre limiti beklenmeden bitiriliyor "
                    f"({(time.time()-t0)/3600:.2f}h / {args.max_hours}h)")

    if IS_MAIN:
        vl = evaluate(model, store, val_b, device)
        best_val = min(best_val, vl)
        commits += 1
        # Son kayıt veri konumunu KORUMALI. Sadece {"final": True} yazmak
        # epoch/batch_pos'u siler ve sonraki oturum veriyi baştan tarar.
        save_ckpt(args.out, model, opt, cfg, step, tokens_seen, pairs_seen,
                  best_val, commits, carry=carry,
                  extra={"final": True, "epoch": epoch, "batch_pos": bi,
                         "max_tokens": max_tokens, "world": WORLD,
                         "hours": (time.time() - t0) / 3600,
                         "stage": stage, "done": done})
        el = (time.time() - t0) / 3600
        log(f"\n[bitti] adım {step} | {pairs_seen/1e6:.2f}M çift | "
            f"{tokens_seen/1e9:.2f}B token ({tokens_seen/chin:.2f}x Chinchilla) | "
            f"val {vl:.4f} | best {best_val:.4f} | {el:.2f}h | -> {args.out}")
        log(f"[durum] {'VERİ BİTTİ — epoch tamam' if done else 'süre/adım limiti — devam gerekiyor'}")
    if use_ddp:
        dist.destroy_process_group()


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
