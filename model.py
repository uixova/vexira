# -*- coding: utf-8 -*-
"""
Vexira — TR<->EN encoder-decoder çeviri modeli.

Mimari: Pre-LN transformer, RMSNorm, SwiGLU, öğrenilmiş mutlak pozisyon,
paylaşılan (tied) embedding, 12 encoder + 6 decoder.

Decoder-only LM'lerden ayrılan bilinçli tercihler:
  - encoder-decoder -> çeviride parametre verimliliği kat kat yüksek
  - öğrenilmiş mutlak pozisyon (RoPE değil): çeviri yeniden-sıralama işi kesin
    pozisyon ister, sabit pencerede RoPE'un extrapolation avantajı gerekmiyor.
    Karşılaştırma için pos_encoding="rope" hâlâ seçilebilir.
    NOT: max_pos=512 tablo boyutudur; ETKİN bağlam 128 token (ön-eğitim
    max_len=128, 128-511 arası satırlar eğitilmedi — bkz. config.py).
  - padding maskeleri ZORUNLU: encoder self-attn ve cross-attn kaynak pad'lerini
    görmemeli
  - label smoothing 0.1 (NMT standardı)

Test:  python model.py test
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VexiraConfig, get_config


# =========================== TEMEL KATMANLAR ===========================

class RMSNorm(nn.Module):
    """Ortalama çıkarmaz, RMS'e böler, fp32'de normalize eder."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dt) * self.weight


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)). d_ff = round64(8/3 * d_model)."""

    def __init__(self, cfg: VexiraConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# =========================== POZİSYON ===========================

def build_sinusoidal(max_pos, dim, device=None):
    """Vaswani 2017 sinusoidal tablosu -> (max_pos, dim)."""
    pos = torch.arange(max_pos, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    inv = torch.exp(-math.log(10000.0) * i / dim)
    pe = torch.zeros(max_pos, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * inv)
    pe[:, 1::2] = torch.cos(pos * inv)
    return pe


def build_rope(max_pos, head_dim, base=10000.0, device=None):
    """RoPE cos/sin tabloları -> her biri (max_pos, head_dim)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                       device=device) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)                     # (max_pos, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)         # (max_pos, head_dim)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, offset=0):
    """q,k: (B, nh, T, hd). cos/sin: (max_pos, hd). offset = KV cache konumu."""
    T = q.size(2)
    c = cos[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)
    s = sin[offset:offset + T].to(q.dtype).unsqueeze(0).unsqueeze(0)
    return q * c + _rotate_half(q) * s, k * c + _rotate_half(k) * s


class Embeddings(nn.Module):
    """Token embedding + pozisyon. lm_head ile tied (ağırlık Vexira'da bağlanır)."""

    def __init__(self, cfg: VexiraConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.scale = math.sqrt(cfg.d_model)

        if cfg.pos_encoding == "learned":
            self.pos = nn.Embedding(cfg.max_pos, cfg.d_model)
        elif cfg.pos_encoding == "sinusoidal":
            # persistent=False -> state_dict'e girmez, resume config'e bağlı kalır
            self.register_buffer("pe", build_sinusoidal(cfg.max_pos, cfg.d_model),
                                 persistent=False)
        # rope: pozisyon burada değil, attention içinde uygulanır

    def forward(self, ids, offset=0):
        """ids: (B, T). offset: KV cache decode adımında mutlak konum."""
        x = self.tok(ids) * self.scale
        T = ids.size(1)
        if self.cfg.pos_encoding == "learned":
            p = torch.arange(offset, offset + T, device=ids.device)
            x = x + self.pos(p).unsqueeze(0)
        elif self.cfg.pos_encoding == "sinusoidal":
            x = x + self.pe[offset:offset + T].unsqueeze(0).to(x.dtype)
        return self.drop(x)


# =========================== ATTENTION ===========================

class Attention(nn.Module):
    """Tek sınıf, üç mod:
       self-attn (encoder)  : is_causal=False, key padding maskesi
       self-attn (decoder)  : is_causal=True,  maske yok (pad hedefler loss'ta zaten
                              ignore_index ile eleniyor)
       cross-attn           : q decoder'dan, k/v encoder memory'den, kaynak pad maskesi
    Attention çekirdeği F.scaled_dot_product_attention.
    """

    def __init__(self, cfg: VexiraConfig, is_cross=False, use_rope=False):
        super().__init__()
        self.nh = cfg.n_head
        self.hd = cfg.head_dim
        self.p = cfg.dropout
        self.is_cross = is_cross
        self.use_rope = use_rope and not is_cross   # cross-attn'e RoPE UYGULANMAZ:
        #   q ve k farklı dizilerden gelir, aralarında ortak pozisyon ekseni yok.
        d = cfg.d_model
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def _split(self, t, B, T):
        return t.view(B, T, self.nh, self.hd).transpose(1, 2)   # (B, nh, T, hd)

    def forward(self, x, mem=None, key_pad=None, is_causal=False,
                rope=None, offset=0):
        """x: (B, Tq, d). mem: cross için (B, S, d). key_pad: (B, S) bool,
        True = GEÇERLİ token (pad DEĞİL)."""
        B, Tq, _ = x.shape
        src = mem if self.is_cross else x
        Tk = src.size(1)

        q = self._split(self.q_proj(x), B, Tq)
        k = self._split(self.k_proj(src), B, Tk)
        v = self._split(self.v_proj(src), B, Tk)

        if self.use_rope and rope is not None:
            q, k = apply_rope(q, k, rope[0], rope[1], offset=offset)

        attn_mask = None
        if key_pad is not None:
            # (B, S) -> (B, 1, 1, S); True = katıl. SDPA is_causal ile birlikte
            # attn_mask kabul etmez, o yüzden decoder self-attn'de key_pad yollamıyoruz.
            attn_mask = key_pad[:, None, None, :]

        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.p if self.training else 0.0)

        o = o.transpose(1, 2).contiguous().view(B, Tq, self.nh * self.hd)
        return self.resid_drop(self.o_proj(o))

    # --- KV cache'li tek-adım decode (yalnız decoder self-attn) ---
    def forward_cached(self, x, cache, pos, rope=None):
        """x: (B, T, d) — prefill'de T>1, decode'da T=1.
        cache: (k_cache, v_cache) her biri (B, nh, max_pos, hd), yerinde yazılır."""
        B, T, _ = x.shape
        q = self._split(self.q_proj(x), B, T)
        k = self._split(self.k_proj(x), B, T)
        v = self._split(self.v_proj(x), B, T)

        if self.use_rope and rope is not None:
            q, k = apply_rope(q, k, rope[0], rope[1], offset=pos)

        kc, vc = cache
        kc[:, :, pos:pos + T] = k
        vc[:, :, pos:pos + T] = v
        k_all = kc[:, :, :pos + T]
        v_all = vc[:, :, :pos + T]

        # prefill (pos==0, T>1) causal olmalı; tek-token decode'da tüm geçmiş görülür
        o = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=(T > 1))
        o = o.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o_proj(o)

    def cross_kv(self, mem):
        """Cross-attn K/V'yi memory'den BİR KEZ hesapla — tüm decode adımlarında
        yeniden kullanılır. Canlı çeviride en büyük tek kazanç."""
        B, S, _ = mem.shape
        return self._split(self.k_proj(mem), B, S), self._split(self.v_proj(mem), B, S)

    def forward_cross_cached(self, x, kv, key_pad=None):
        B, T, _ = x.shape
        q = self._split(self.q_proj(x), B, T)
        k, v = kv
        attn_mask = key_pad[:, None, None, :] if key_pad is not None else None
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)
        return self.o_proj(o)


# =========================== BLOKLAR ===========================

class EncoderLayer(nn.Module):
    def __init__(self, cfg: VexiraConfig):
        super().__init__()
        use_rope = cfg.pos_encoding == "rope"
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = Attention(cfg, is_cross=False, use_rope=use_rope)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, key_pad=None, rope=None):
        x = x + self.self_attn(self.norm1(x), key_pad=key_pad,
                               is_causal=False, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: VexiraConfig):
        super().__init__()
        use_rope = cfg.pos_encoding == "rope"
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.self_attn = Attention(cfg, is_cross=False, use_rope=use_rope)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.cross_attn = Attention(cfg, is_cross=True)
        self.norm3 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, mem, src_pad=None, rope=None):
        x = x + self.self_attn(self.norm1(x), is_causal=True, rope=rope)
        x = x + self.cross_attn(self.norm2(x), mem=mem, key_pad=src_pad)
        x = x + self.mlp(self.norm3(x))
        return x

    def forward_step(self, x, self_cache, cross_kv, pos, src_pad=None, rope=None):
        x = x + self.self_attn.forward_cached(self.norm1(x), self_cache, pos, rope=rope)
        x = x + self.cross_attn.forward_cross_cached(self.norm2(x), cross_kv,
                                                     key_pad=src_pad)
        x = x + self.mlp(self.norm3(x))
        return x


# =========================== KAYIP ===========================

def chunked_ce(logits_fn, hidden, targets, vocab_size, label_smoothing=0.1,
               ignore_index=-100, chunk=256):
    """Label smoothing'li, parçalı cross-entropy.

    hidden: (N, d) düzleştirilmiş decoder çıktısı. logits_fn: hidden -> logits.
    Logit tensörünü (N, 32000) tek seferde üretmek yerine parça parça üretip
    kaybı toplar -> peak bellek %60-70 düşer. fp32 akümülatör, tam maskeli
    parçada NaN üretmez.
    """
    total = hidden.new_zeros((), dtype=torch.float32)
    n_valid = 0
    eps = label_smoothing
    for i in range(0, hidden.size(0), chunk):
        h = hidden[i:i + chunk]
        t = targets[i:i + chunk]
        keep = t != ignore_index
        cnt = int(keep.sum())
        if cnt == 0:
            continue
        logits = logits_fn(h[keep]).float()
        logp = F.log_softmax(logits, dim=-1)
        nll = -logp.gather(1, t[keep].unsqueeze(1)).squeeze(1)
        if eps > 0.0:
            smooth = -logp.mean(dim=-1)
            loss = (1.0 - eps) * nll + eps * smooth
        else:
            loss = nll
        total = total + loss.sum()
        n_valid += cnt
    if n_valid == 0:
        # Tamamen maskeli batch: 0 döndür ama grafiği kopar(ma) — grad akışı için
        # hidden'a bağlı sıfır üret.
        return hidden.sum() * 0.0
    return total / n_valid


# =========================== MODEL ===========================

class Vexira(nn.Module):
    """Encoder-decoder çeviri modeli.

    forward(...) EĞİTİMDE sadece skaler loss döndürür — DataParallel'in çıktıyı
    GPU'lar arası dengeleyebilmesi için.
    """

    def __init__(self, cfg: VexiraConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layer)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layer)])
        self.enc_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.dec_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.tok.weight

        if cfg.pos_encoding == "rope":
            cos, sin = build_rope(cfg.max_pos, cfg.head_dim, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        # train.py tarafından ayarlanır: ilk N bloğa gradient checkpointing
        self.grad_ckpt_enc = 0
        self.grad_ckpt_dec = 0
        self.amp_dtype = None      # train.py fp16/bf16 seçince atanır

        self.apply(self._init)
        # Rezidüel-ölçekli init: derinlikle büyüyen varyansı bastır
        n_layer = cfg.n_enc_layer + cfg.n_dec_layer
        std = 0.02 / math.sqrt(2 * n_layer)
        for m in self.modules():
            if isinstance(m, Attention):
                nn.init.normal_(m.o_proj.weight, mean=0.0, std=std)
            elif isinstance(m, SwiGLU):
                nn.init.normal_(m.down_proj.weight, mean=0.0, std=std)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @property
    def _rope(self):
        if self.cfg.pos_encoding != "rope":
            return None
        return (self.rope_cos, self.rope_sin)

    def num_params(self, non_embedding=False):
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) in seen:      # tied ağırlığı iki kez sayma
                continue
            seen.add(id(p))
            total += p.numel()
        if non_embedding:
            total -= self.embed.tok.weight.numel()
        return total

    # ----------------------------- ENCODE -----------------------------
    def encode(self, src, src_pad=None):
        """src: (B, S) token id. src_pad: (B, S) bool, True = geçerli token."""
        x = self.embed(src)
        rope = self._rope
        for i, layer in enumerate(self.encoder):
            if self.training and i < self.grad_ckpt_enc:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, src_pad, rope, use_reentrant=False)
            else:
                x = layer(x, key_pad=src_pad, rope=rope)
        return self.enc_norm(x)

    # ----------------------------- DECODE -----------------------------
    def decode(self, tgt_in, mem, src_pad=None):
        """tgt_in: (B, T) — teacher forcing girdisi (<bos> ile başlar)."""
        x = self.embed(tgt_in)
        rope = self._rope
        for i, layer in enumerate(self.decoder):
            if self.training and i < self.grad_ckpt_dec:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, mem, src_pad, rope, use_reentrant=False)
            else:
                x = layer(x, mem, src_pad=src_pad, rope=rope)
        return self.dec_norm(x)

    # ----------------------------- FORWARD -----------------------------
    def forward(self, src, tgt_in, tgt_out=None, src_pad=None, ignore_index=-100):
        """tgt_out verilirse SKALER loss, yoksa logits döner."""
        ctx = (torch.autocast(device_type=src.device.type, dtype=self.amp_dtype)
               if self.amp_dtype is not None else _NullCtx())
        with ctx:
            mem = self.encode(src, src_pad)
            h = self.decode(tgt_in, mem, src_pad)
            if tgt_out is None:
                return self.lm_head(h)
            loss = chunked_ce(
                self.lm_head, h.reshape(-1, h.size(-1)), tgt_out.reshape(-1),
                self.cfg.vocab_size, label_smoothing=self.cfg.label_smoothing,
                ignore_index=ignore_index)
        return loss

    # --------------------- KV CACHE'Lİ ÇIKARIM ---------------------
    def init_cache(self, mem, max_len=None):
        """Decoder self-attn cache'leri + cross-attn K/V'leri hazırla.
        Cross K/V BİR KEZ hesaplanır, tüm decode adımlarında yeniden kullanılır."""
        B = mem.size(0)
        L = max_len or self.cfg.max_pos
        nh, hd = self.cfg.n_head, self.cfg.head_dim
        dev, dt = mem.device, mem.dtype
        self_caches, cross_kvs = [], []
        for layer in self.decoder:
            self_caches.append((torch.zeros(B, nh, L, hd, device=dev, dtype=dt),
                                torch.zeros(B, nh, L, hd, device=dev, dtype=dt)))
            cross_kvs.append(layer.cross_attn.cross_kv(mem))
        return self_caches, cross_kvs

    @torch.no_grad()
    def decode_step(self, tgt_ids, caches, pos, src_pad=None):
        """tgt_ids: (B, T) — prefill'de T>1 (pos=0), sonra T=1.
        Dönen: (B, T, vocab) logits."""
        self_caches, cross_kvs = caches
        x = self.embed(tgt_ids, offset=pos)
        rope = self._rope
        for layer, sc, ck in zip(self.decoder, self_caches, cross_kvs):
            x = layer.forward_step(x, sc, ck, pos, src_pad=src_pad, rope=rope)
        return self.lm_head(self.dec_norm(x))


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# =========================== BİRİM TESTLER ===========================

def _test():
    torch.manual_seed(0)
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
        ok = ok and cond

    print("\n[1] Parametre sayıları config ile uyuşuyor mu")
    for name in ("tiny", "main"):
        cfg = get_config(name)
        m = Vexira(cfg)
        got = m.num_params()
        want = cfg.param_breakdown()["total"]
        check(f"{name}: {got/1e6:.2f}M (beklenen {want/1e6:.2f}M)",
              abs(got - want) < 1000, f"fark={got-want}")
        del m

    cfg = get_config("tiny", dropout=0.0)
    model = Vexira(cfg).eval()
    B, S, T = 2, 7, 5
    src = torch.randint(4, 100, (B, S))
    tgt = torch.randint(4, 100, (B, T))

    print("\n[2] Şekiller")
    logits = model(src, tgt)
    check("logits (B,T,V)", tuple(logits.shape) == (B, T, cfg.vocab_size),
          str(tuple(logits.shape)))
    # DİKKAT: teacher forcing kayması şart. tgt_in = <bos> + tgt[:-1].
    # tgt_in = tgt_out verilirse pozisyon t kendi hedefini görür ve tied embedding
    # üzerinden kopyalar -> loss yapay olarak ln(V)'nin çok altına iner.
    tgt_in = torch.cat([torch.full((B, 1), 2), tgt[:, :-1]], dim=1)
    loss = model(src, tgt_in, tgt)
    check("loss skaler", loss.dim() == 0, f"{loss.item():.4f}")
    check("loss ~ ln(V) civarı (rastgele init)", 8.0 < loss.item() < 12.0,
          f"{loss.item():.3f} vs ln(V)={math.log(cfg.vocab_size):.3f}")
    leak = model(src, tgt, tgt).item()
    check("kaydırılmamış girdi loss'u BELİRGİN düşük (sızıntı kanıtı, beklenen)",
          leak < loss.item() - 1.0, f"kaydırmasız={leak:.3f} < kaydırmalı={loss.item():.3f}")

    print("\n[3] Decoder causal — geleceğe sızıntı YOK")
    # tgt'nin SON token'ını değiştir; t<T-1 konumlarındaki logits DEĞİŞMEMELİ
    with torch.no_grad():
        a = model(src, tgt)
        tgt2 = tgt.clone()
        tgt2[:, -1] = (tgt2[:, -1] + 7) % 100
        b = model(src, tgt2)
    check("t<T-1 logits sabit", torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5),
          f"maxdiff={((a[:, :-1]-b[:, :-1]).abs().max()):.2e}")
    check("t=T-1 logits değişti", not torch.allclose(a[:, -1], b[:, -1], atol=1e-5))

    print("\n[4] Cross-attn kaynağın TAMAMINI görüyor")
    with torch.no_grad():
        src2 = src.clone()
        src2[:, 0] = (src2[:, 0] + 13) % 100     # ilk kaynak token'ı değiştir
        c = model(src2, tgt)
    check("ilk kaynak token'ı değişince İLK hedef logit'i de değişti",
          not torch.allclose(a[:, 0], c[:, 0], atol=1e-5))

    print("\n[5] Kaynak padding maskesi işliyor")
    with torch.no_grad():
        src_pad = torch.ones(B, S, dtype=torch.bool)
        src_pad[:, -2:] = False                   # son 2 token pad
        d1 = model(src, tgt, src_pad=src_pad)
        src3 = src.clone()
        src3[:, -2:] = (src3[:, -2:] + 21) % 100  # SADECE pad bölgeyi değiştir
        d2 = model(src3, tgt, src_pad=src_pad)
    check("maskeli kaynak token'ları çıktıyı etkilemiyor",
          torch.allclose(d1, d2, atol=1e-5),
          f"maxdiff={((d1-d2).abs().max()):.2e}")

    print("\n[6] KV cache == cache'siz (greedy yol)")
    with torch.no_grad():
        full = model(src, tgt)                          # cache'siz
        mem = model.encode(src, None)
        caches = model.init_cache(mem, max_len=cfg.max_pos)
        step0 = model.decode_step(tgt[:, :1], caches, 0)          # prefill T=1
        outs = [step0]
        for t in range(1, T):
            outs.append(model.decode_step(tgt[:, t:t + 1], caches, t))
        cached = torch.cat(outs, dim=1)
    check("cache'li logits == cache'siz logits",
          torch.allclose(full, cached, atol=1e-4),
          f"maxdiff={((full-cached).abs().max()):.2e}")

    print("\n[7] Prefill (T>1) + tek-adım karışık kullanım")
    with torch.no_grad():
        mem = model.encode(src, None)
        caches = model.init_cache(mem, max_len=cfg.max_pos)
        pre = model.decode_step(tgt[:, :3], caches, 0)            # prefill 3 token
        nxt = model.decode_step(tgt[:, 3:4], caches, 3)
        mixed = torch.cat([pre, nxt], dim=1)
    check("prefill+decode == cache'siz (ilk 4 adım)",
          torch.allclose(full[:, :4], mixed, atol=1e-4),
          f"maxdiff={((full[:, :4]-mixed).abs().max()):.2e}")

    print("\n[8] chunked_ce label smoothing doğruluğu")
    h = torch.randn(64, cfg.d_model)
    t = torch.randint(0, cfg.vocab_size, (64,))
    head = model.lm_head
    got = chunked_ce(head, h, t, cfg.vocab_size, label_smoothing=0.1, chunk=16)
    ref = F.cross_entropy(head(h).float(), t, label_smoothing=0.1)
    check("chunked_ce == F.cross_entropy(label_smoothing=0.1)",
          torch.allclose(got, ref, atol=1e-4), f"{got.item():.6f} vs {ref.item():.6f}")

    t_masked = t.clone()
    t_masked[:32] = -100
    got_m = chunked_ce(head, h, t_masked, cfg.vocab_size, label_smoothing=0.1, chunk=16)
    ref_m = F.cross_entropy(head(h[32:]).float(), t[32:], label_smoothing=0.1)
    check("ignore_index doğru elenmiş", torch.allclose(got_m, ref_m, atol=1e-4),
          f"{got_m.item():.6f} vs {ref_m.item():.6f}")

    all_masked = torch.full_like(t, -100)
    got_z = chunked_ce(head, h, all_masked, cfg.vocab_size, chunk=16)
    check("tamamen maskeli batch NaN değil", torch.isfinite(got_z).all(),
          f"{got_z.item()}")

    print("\n[9] pos_encoding varyantları çalışıyor")
    for pe in ("learned", "sinusoidal", "rope"):
        c = get_config("tiny", pos_encoding=pe, dropout=0.0)
        m = Vexira(c).eval()
        with torch.no_grad():
            f_ = m(src, tgt)
            mm = m.encode(src, None)
            cc = m.init_cache(mm, max_len=c.max_pos)
            o = torch.cat([m.decode_step(tgt[:, i:i + 1], cc, i) for i in range(T)], 1)
        check(f"{pe}: forward + KV cache tutarlı",
              torch.allclose(f_, o, atol=1e-4),
              f"maxdiff={((f_-o).abs().max()):.2e}")

    print("\n[10] Gradient akışı + gradient checkpointing")
    m = Vexira(get_config("tiny")).train()
    m.grad_ckpt_enc, m.grad_ckpt_dec = 2, 1
    l = m(src, tgt, tgt.clone())
    l.backward()
    n_none = sum(1 for p in m.parameters() if p.grad is None)
    g = m.embed.tok.weight.grad
    check("tüm parametrelerde grad var", n_none == 0, f"grad'sız={n_none}")
    check("embed grad sıfır değil (tied)", g is not None and g.abs().sum() > 0)

    print("\n" + ("HEPSİ GEÇTİ ✓" if ok else "BAŞARISIZ ✗"))
    return ok


def _overfit():
    """Ezberleme testi: 200 sentetik çifti ezberle. Loss ~0'a inmezse mimaride bug var."""
    torch.manual_seed(0)
    cfg = get_config("tiny", vocab_size=512, max_pos=32, dropout=0.0, label_smoothing=0.0)
    model = Vexira(cfg).train()
    N, S, T = 200, 8, 8
    src = torch.randint(4, 512, (N, S))
    tgt = torch.randint(4, 512, (N, T))
    tgt_in = torch.cat([torch.full((N, 1), 2), tgt[:, :-1]], dim=1)   # <bos>=2
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.98))

    # Hedefler tamamen rastgele token (yapı yok) -> saf ezber. Yeterli adım + LR
    # sönümü şart; erken kesilirse "mimaride bug" gibi görünür ama değildir.
    steps = 1200
    print(f"\nEzberleme testi: {N} çift, {model.num_params()/1e6:.2f}M param, {steps} adım")
    for step in range(1, steps + 1):
        for g in opt.param_groups:                       # cosine 3e-3 -> 3e-5
            g["lr"] = 3e-5 + 0.5 * (3e-3 - 3e-5) * (1 + math.cos(math.pi * step / steps))
        i = torch.randint(0, N, (32,))
        loss = model(src[i], tgt_in[i], tgt[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            print(f"  adım {step:4d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        final = model(src, tgt_in, tgt).item()
        pred = model(src, tgt_in).argmax(-1)
        acc = (pred == tgt).float().mean().item()
    print(f"\n  final loss = {final:.4f}   token doğruluğu = {acc*100:.1f}%")
    ok = final < 0.15 and acc > 0.97
    print("  " + ("EZBERLEME GEÇTİ ✓" if ok else "EZBERLEME BAŞARISIZ ✗ — mimaride bug var"))
    return ok


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "test":
        raise SystemExit(0 if _test() else 1)
    if arg == "overfit":
        raise SystemExit(0 if _overfit() else 1)
    print("kullanım: python model.py test | overfit")
