# -*- coding: utf-8 -*-
"""
Vexira — TR<->EN çift yönlü çeviri modeli. Mimari konfigürasyonu.

Hparam'lar modül-seviyesi global DEĞİL, dataclass. Sebep: encoder-decoder'da iki
ayrı derinlik var (n_enc_layer / n_dec_layer) ve bunları checkpoint'ten geri
yükleyip modül global'lerine monkey-patch etmek hataya çok açık. Config
checkpoint'e olduğu gibi yazılır, olduğu gibi geri okunur.
"""

from dataclasses import dataclass, asdict, fields


ARCH = "vexira_encdec"     # checkpoint kilidi — yabancı ckpt asla yüklenemesin


def _round64(x):
    """FFN gizli boyutunu 64'ün katına yuvarla (GEMM tile hizası)."""
    return 64 * ((int(x) + 63) // 64)


@dataclass
class VexiraConfig:
    # --- boyut ---
    vocab_size: int = 32000        # ortak TR+EN SentencePiece unigram
    d_model: int = 512
    n_head: int = 8                # head_dim = 512/8 = 64
    n_enc_layer: int = 12          # anlam/bağlam çözümü burada olur
    n_dec_layer: int = 6           # Türkçe çekim eki zinciri için derinlik
    d_ff: int = 0                  # 0 -> __post_init__ round64(8/3*d_model) hesaplar
    # Pozisyon TABLOSUNUN boyutu — KULLANILABİLİR BAĞLAM DEĞİL.
    # Ön-eğitim max_len=128 ile yapıldı; 128-511 arası satırlar hiç güncellenmedi
    # (ölçüm: norm 9.5-10.4 vs 0.34). Etkin bağlam 128 token.
    # DEĞİŞTİRME: checkpoint'teki tensör (512,512) ve matches() buna bakıyor;
    # 128 yazmak resume'u "config uyuşmuyor" diye sıfırdan başlatır.
    # Çıkarımda gerçek tavan translate.trained_positions() ile ÖLÇÜLÜR.
    max_pos: int = 512

    # --- regularizasyon ---
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # --- sayısal ---
    rms_eps: float = 1e-6
    rope_base: float = 10000.0     # yalnız pos_encoding="rope" iken kullanılır

    # --- mimari anahtarları ---
    pos_encoding: str = "learned"  # learned | sinusoidal | rope  (Faz 1a A/B testi)
    tie_embeddings: bool = True    # embed <-> lm_head paylaşımı (16.4M tasarruf, zorunlu)

    # --- kimlik ---
    arch: str = ARCH

    def __post_init__(self):
        if self.d_ff == 0:
            # Llama oranı: 8/3 * d, 64'e yuvarlanmış. d=512 -> 1408
            self.d_ff = _round64(8 / 3 * self.d_model)
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) n_head'e ({self.n_head}) bölünmeli")
        if self.pos_encoding not in ("learned", "sinusoidal", "rope"):
            raise ValueError(f"bilinmeyen pos_encoding: {self.pos_encoding}")
        if self.pos_encoding == "rope" and (self.d_model // self.n_head) % 2 != 0:
            raise ValueError("rope head_dim'in çift olmasını ister")

    # --- parametre muhasebesi (ölçüm için, model kurmadan) ---
    @property
    def head_dim(self):
        return self.d_model // self.n_head

    def param_breakdown(self):
        """Katman başına parametre dökümü. Model kurmadan bütçe kontrolü için."""
        d, f = self.d_model, self.d_ff
        attn = 4 * d * d                     # q,k,v,o (bias yok)
        ffn = 3 * d * f                      # SwiGLU: gate, up, down
        enc_layer = attn + ffn
        dec_layer = attn + attn + ffn        # self + cross + ffn
        embed = self.vocab_size * d          # tied ise bir kez sayılır
        pos = self.max_pos * d if self.pos_encoding == "learned" else 0
        # RMSNorm ağırlıkları: enc 2/katman, dec 3/katman, + 2 final norm
        norms = d * (2 * self.n_enc_layer + 3 * self.n_dec_layer + 2)
        out = {
            "embed": embed,
            "pos": pos,
            "encoder": enc_layer * self.n_enc_layer,
            "decoder": dec_layer * self.n_dec_layer,
            "norms": norms,
        }
        if not self.tie_embeddings:
            out["lm_head"] = embed
        out["total"] = sum(out.values())
        return out

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Checkpoint'ten geri yükle. Bilinmeyen anahtarları yok say (ileri uyumluluk)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def matches(self, other):
        """Resume güvenliği: ağırlık şeklini etkileyen alanlar birebir aynı mı?
        dropout/label_smoothing gibi ağırlık şeklini değiştirmeyenler serbest."""
        keys = ("arch", "vocab_size", "d_model", "n_head", "n_enc_layer",
                "n_dec_layer", "d_ff", "max_pos", "pos_encoding", "tie_embeddings")
        o = other if isinstance(other, dict) else other.to_dict()
        s = self.to_dict()
        return all(s[k] == o.get(k) for k in keys)


# --------------------------- HAZIR CONFIG'LER ---------------------------

def main_config(**kw):
    """Üretim modeli — ~80.5M parametre. Tam eğitim uzak sunucuda."""
    return VexiraConfig(**kw)


def tiny_config(**kw):
    """Yerel debug — ~11M. Ezberleme testi ve pos_encoding A/B için."""
    base = dict(vocab_size=32000, d_model=256, n_head=4,
                n_enc_layer=4, n_dec_layer=2, max_pos=256)
    base.update(kw)
    return VexiraConfig(**base)


PRESETS = {"tiny": tiny_config, "main": main_config}


def get_config(name="main", **kw):
    if name not in PRESETS:
        raise KeyError(f"bilinmeyen preset: {name} (seçenekler: {list(PRESETS)})")
    return PRESETS[name](**kw)


if __name__ == "__main__":
    for name in ("tiny", "main"):
        cfg = get_config(name)
        bd = cfg.param_breakdown()
        print(f"\n=== {name} === d={cfg.d_model} ff={cfg.d_ff} "
              f"enc={cfg.n_enc_layer} dec={cfg.n_dec_layer} vocab={cfg.vocab_size}")
        for k, v in bd.items():
            if k == "total":
                continue
            print(f"  {k:9s} {v/1e6:8.2f}M  ({100*v/bd['total']:5.1f}%)")
        print(f"  {'TOPLAM':9s} {bd['total']/1e6:8.2f}M")
