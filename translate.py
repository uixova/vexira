# -*- coding: utf-8 -*-
"""
Vexira çıkarım — beam search + KV cache.

TAŞINABİLİR: saf metin -> metin. İşletim sistemine, masaüstü ortamına, ekran
sunucusuna ya da harici servise HİÇBİR bağımlılığı yok. Yalnızca Python + torch.
Ekran yakalama, OCR, TTS/STT gibi sistem işleri BİLİNÇLİ olarak dışarıda:
onları çağıran script yapar, buraya sadece metin gelir.

  [ekran bölgesi seçen script] -> [OCR] -> METİN -> translate() -> METİN -> [overlay]
                                                     ^^^^^^^^^^^
                                                  bu dosyanın tamamı

SIKIŞTIRMA YOK. int8/ternary yolu bilinçli olarak yok: bu ölçekte kalite kaybı
maliyetine değmiyor. Hız kaynakları:
  - encoder BİR KEZ koşar, cross-attn K/V tüm decode adımlarında yeniden kullanılır
  - decoder self-attn KV cache
  - toplu (batch) çeviri + uzunluk kovalama
  - beam 2 (beam 4 ~+0.5 BLEU ama 2x yavaş — canlı kullanımda 2 yeterli)

Ağırlık fp16 saklanır (~161 MB); CPU'da fp32'ye çevrilip hesaplanır çünkü çoğu
masaüstü/dizüstü CPU'sunda fp16 GEMM donanımı yok, fp32 daha hızlı. Bu davranış
--fp32-store ile kapatılabilir.

Kullanım:
  python translate.py --text "Hello world" --to tr
  python translate.py --file lines.txt --to tr --domain ocr
  python translate.py --srt film.srt --to tr --out film.tr.srt
  python translate.py --server                    # stdin/stdout JSON, satır satır
"""

import argparse
import json
import os
import re
import sys
import time

import torch

from config import VexiraConfig
from glossary import Glossary
from model import Vexira
from tokenizer import VexiraTokenizer, DEFAULT_MODEL as SPM_DEFAULT

PAD_ID, BOS_ID, EOS_ID = 0, 2, 3
# Ana model ince ayarlı olan. Ölçüm (559 satırlık gerçek Ren'Py dosyası):
#   vexira.pt      33 tutarsız terim · FLORES en->tr 29.53 / tr->en 35.03
#   vexira_sft.pt  19 tutarsız terim · FLORES en->tr 29.13 / tr->en 35.16
# Terim tutarlılığı %42 düzelirken genel çeviri kaybı gürültü içinde.
DEFAULT_CKPT = "models/vexira_sft.pt"

# Çıktı onarımı — ölçülen iki gerçek kusur (559 satırlık .rpy testi):
#   1) "\n" -> "\ n"      SentencePiece kaçış dizisini iki parçaya bölüyor,
#                         decode araya boşluk koyuyor. Ren'Py'de satır sonu ölür.
#   2) "[text]" -> "[metin]"  model değişken ADINI çeviriyor; motor o adı arar,
#                         bulamaz, oyun bozulur.
# İkisi de HER ZAMAN hatadır (geçerli bir çıktı bu biçimde olamaz), o yüzden
# onarım güvenli ve varsayılan olarak açık.
_ESCAPE_FIX = re.compile(r"\\\s+([nrt])")
_PH_OUT = re.compile(r"(\{[^{}]*\}|%[-0-9.]*[A-Za-z]|\[[^\[\]]*\])")


def repair_output(src, out):
    """Kaçış dizilerini birleştir, yer tutucuları kaynaktakiyle eşitle."""
    out = _ESCAPE_FIX.sub(r"\\\1", out)
    a, b = _PH_OUT.findall(src), _PH_OUT.findall(out)
    # Yalnız SAYI ve SIRA aynıysa konumsal geri yazma yapılır. Sayı tutmuyorsa
    # model bir tanesini tamamen düşürmüş demektir; hangisinin nereye geleceği
    # belirsiz olduğu için dokunulmaz (yanlış yere koymak sessiz bozulmadır).
    if a and len(a) == len(b) and a != b:
        it = iter(a)
        out = _PH_OUT.sub(lambda m: next(it), out)
    return out


# --------------------------- ÇALIŞMA AYARLARI ---------------------------
#
# Bu ayarlar MODELİN İÇİNE gömülür (ckpt["runtime"]). Sebebi: doğru değerler
# modele özgü ve ölçümle bulundu; kullanıcının bunları bilmesi ya da doğru
# tahmin etmesi beklenemez. Model nereye giderse ayarı da yanında gider.
#
# Öncelik sırası (üstteki kazanır):
#   1. CLI bayrağı           --threads 8
#   2. Ortam değişkeni       VEXIRA_THREADS=8
#   3. Checkpoint'e gömülü   ckpt["runtime"]["threads"]
#   4. Buradaki varsayılan
#
# thread neden 4: bu boyutta (80M, d_model 512) katman başına iş küçük,
# senkronizasyon maliyeti hesabı geçiyor. 12 çekirdekli makinede ölçüldü:
#     thread= 4, beam=2 ->  77.6 ms/satır   (12.9 satır/sn)
#     thread=12, beam=2 -> 466.0 ms/satır   ( 2.1 satır/sn)  <- torch varsayılanı
# Varsayılanı torch'a bırakmak 6x yavaş demek.
RUNTIME_DEFAULTS = {
    "threads": 4,
    "beam": 2,
    "batch_size": 32,
    "domain": "sub",
}


def _env_int(name):
    v = os.environ.get(name)
    try:
        return int(v) if v else None
    except ValueError:
        return None


def resolve_runtime(ckpt_runtime=None, **cli):
    """Gömülü ayar + ortam + CLI -> son değerler."""
    r = dict(RUNTIME_DEFAULTS)
    r.update({k: v for k, v in (ckpt_runtime or {}).items() if v is not None})
    for key, env in (("threads", "VEXIRA_THREADS"), ("beam", "VEXIRA_BEAM"),
                     ("batch_size", "VEXIRA_BATCH")):
        v = _env_int(env)
        if v:
            r[key] = v
    if os.environ.get("VEXIRA_DOMAIN"):
        r["domain"] = os.environ["VEXIRA_DOMAIN"]
    r.update({k: v for k, v in cli.items() if v})
    r["threads"] = max(1, min(int(r["threads"]), os.cpu_count() or 1))
    return r


def default_threads():
    return resolve_runtime()["threads"]


def trained_positions(pos_weight, floor=0.25):
    """Öğrenilmiş pozisyon gömmelerinden KAÇ TANESİ gerçekten eğitilmiş?

    config.max_pos=512 diyor ama ön-eğitim max_len=128 ile yapıldıysa 128-511
    arası satırlar ilk değerinde kalır. Ölçüldü (bu checkpoint):
        pos   0-127 : norm 9.5 - 10.4     eğitilmiş
        pos 128-511 : norm 0.34           dokunulmamış
    O bölgeyi kullanmak çıktıyı bozuyor — 143 tokenlik bir girdide ilk cümle
    tamamen kayboldu ve model tekrara girdi.

    Eşik: ilk 32 satırın ortalama normunun `floor` katı. Sabit sayı yazmak
    yerine ölçmek, ileride 512'ye kadar eğitilirse kodun kendiliğinden
    genişlemesini sağlar.
    """
    n = pos_weight.detach().norm(dim=-1).float()
    ref = float(n[:32].mean()) * floor
    live = (n > ref).nonzero().flatten()
    return int(live[-1]) + 1 if len(live) else int(pos_weight.shape[0])


_SENT_SPLIT = re.compile(r"(?<=[.!?…:;])\s+")

# Bölünen parçalara verilecek <ctx> için ayrılan token bütçesi.
# Parçalar bu kadar daha kısa kesilir; toplam yine tavanın altında kalır.
CTX_ALLOW = 48


def split_to_budget(text, measure, budget):
    """Metni, her parçası `budget` token'ı geçmeyecek şekilde böl.

    Kırpmak metni SESSİZCE siler; bölmek içeriği korur. Önce cümle sınırından,
    cümle de sığmıyorsa kelime sınırından bölünür.
    """
    if measure(text) <= budget:
        return [text]
    out, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        cand = (cur + " " + sent).strip() if cur else sent
        if cur and measure(cand) > budget:
            out.append(cur)
            cur = sent
        else:
            cur = cand
    if cur:
        out.append(cur)

    final = []
    for part in out:
        if measure(part) <= budget:
            final.append(part)
            continue
        words, buf = part.split(), ""
        for w in words:
            cand = (buf + " " + w).strip() if buf else w
            if buf and measure(cand) > budget:
                final.append(buf)
                buf = w
            else:
                buf = cand
        if buf:
            final.append(buf)
    return final or [text]


def resolve_spm(ckpt_path, explicit=None):
    """Tokenizer dosyasını sırayla ara. Modeli başka bir makineye kopyalarken
    veri havuzu yolu bulunmayabilir; spm'yi checkpoint'in yanına koymak yeter.

      1. --spm ile verilen yol
      2. VEXIRA_SPM ortam değişkeni
      3. checkpoint ile aynı klasör  (taşınabilir kurulum)
      4. paket varsayılanı (veri havuzu)
    """
    # Açıkça verilen yol YOKSA sessizce başkasına düşme: yanlış tokenizer
    # çöp çıktı üretir ve bu hata çalışma anında fark edilmez.
    if explicit and not os.path.exists(explicit):
        raise FileNotFoundError(f"--spm ile verilen dosya yok: {explicit}")

    cands = [explicit, os.environ.get("VEXIRA_SPM")]
    if ckpt_path:
        d = os.path.dirname(os.path.abspath(ckpt_path))
        cands += [os.path.join(d, "vexira_spm.model"),
                  os.path.join(d, "spm", "vexira_spm.model")]
    cands.append(SPM_DEFAULT)
    for c in cands:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "tokenizer bulunamadı. Denenen yollar:\n  "
        + "\n  ".join(str(c) for c in cands if c)
        + "\n--spm ile yol ver ya da vexira_spm.model'i checkpoint'in yanına koy.")


class Translator:
    def __init__(self, ckpt=DEFAULT_CKPT, spm=None, device="cpu",
                 fp16_store=True, glossary=None, use_glossary=True,
                 repair=True, inject=False):
        self.device = torch.device(device)
        self.repair = repair
        self.tok = VexiraTokenizer(resolve_spm(ckpt, spm))

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = VexiraConfig.from_dict(ck["config"])
        if self.cfg.vocab_size != self.tok.vocab_size:
            raise ValueError(
                f"vocab uyuşmazlığı: model {self.cfg.vocab_size}, "
                f"tokenizer {self.tok.vocab_size} — yanlış spm dosyası?")
        self.cfg.dropout = 0.0
        self.model = Vexira(self.cfg)
        self.model.load_state_dict(ck["model"])
        # Depolama fp16, hesap fp32 (CPU'da fp16 kernel yok)
        if fp16_store and self.device.type == "cpu":
            self.model.half().float()
        self.model.to(self.device).eval()

        # Gerçek pozisyon tavanı ölçülür, config'e GÜVENİLMEZ (bkz. yukarıda).
        self.max_pos_used = min(self.cfg.max_pos,
                                trained_positions(self.model.embed.pos.weight))
        if self.max_pos_used < self.cfg.max_pos:
            print(f"[uyarı] pozisyon gömmelerinin yalnız {self.max_pos_used}/"
                  f"{self.cfg.max_pos} tanesi eğitilmiş — uzun girdiler cümle "
                  f"sınırından bölünecek", file=sys.stderr)

        # --- sözlük: dosyadan verilen kazanır, yoksa checkpoint'e gömülü olan ---
        # Modeli tek dosya taşımak mümkün olsun diye sözlük ckpt'nin içinde durur
        # (bkz. glossary.py). Harici TSV verilirse onu ezer.
        # inject: cümle içi terim enjeksiyonu. Varsayılan KAPALI — ön-eğitilmiş
        # model koruma bölgesine güvenilir biçimde uymuyor (bkz. glossary.py).
        # İnce ayar sonrası --inject ile açılıp yeniden ölçülecek.
        if not use_glossary:
            self.gloss = Glossary([])
        elif glossary:
            self.gloss = (glossary if isinstance(glossary, Glossary)
                          else Glossary.from_tsv(glossary))
            self.gloss.allow_inject = inject
            self.gloss._build()
        else:
            self.gloss = Glossary.from_dict(ck.get("glossary"))
            self.gloss.allow_inject = inject

        # Modele gömülü çalışma ayarları (thread/beam/batch). Yoksa varsayılan.
        self.runtime = resolve_runtime(ck.get("runtime"))
        self.info = {"step": ck.get("step"), "best_val": ck.get("best_val"),
                     "params": self.model.num_params(),
                     "glossary": len(self.gloss),
                     "runtime": self.runtime}
        self.stats = {"exact": 0, "injected": 0, "model": 0, "split": 0}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def translate(self, lines, tgt_lang="tr", ctx=None, domain="sub",
                  beam=2, max_ratio=2.0, max_new=None, len_penalty=1.0,
                  batch_size=32):
        """CANLI PIPELINE ARAYÜZÜ — OCR/STT katmanı yalnız bunu çağırır.

        lines : çevrilecek satırlar
        ctx   : satır başına önceki satır listesi (ya da tüm liste için ortak)
        """
        if isinstance(lines, str):
            lines = [lines]
        if not lines:
            return []

        out = [None] * len(lines)
        encoded, todo = [], []          # todo: modele gidecek satırların indeksi

        for i, ln in enumerate(lines):
            # --- katman 1: TAM EŞLEŞME. Model hiç çağrılmaz, sonuç deterministik.
            # Arayüz etiketlerinin tamamı buraya düşer; tutarsızlık matematiksel
            # olarak imkânsız hâle gelir.
            if len(self.gloss):
                hit = self.gloss.lookup(ln, tgt_lang, domain=domain)
                if hit is not None:
                    out[i] = hit
                    self.stats["exact"] += 1
                    continue

            # --- katman 2: CÜMLE İÇİ ENJEKSİYON. Terim hedef karşılığıyla
            # değiştirilip koruma bölgesine alınır; model kalanı çevirir ve
            # gerekirse Türkçe ekini terimin sonuna ekler.
            text, keep = ln, None
            if len(self.gloss):
                text, keep, n = self.gloss.rewrite(ln, tgt_lang, domain=domain)
                if n:
                    self.stats["injected"] += 1

            c = None
            if ctx:
                c = ctx[i] if isinstance(ctx[0], (list, tuple)) else ctx

            # --- katman 3: UZUN GİRDİ BÖLME.
            # Eğitilmemiş pozisyonlara taşan girdi çöp üretiyor (ölçüldü: 143
            # tokenlik cümlede ilk yarı kayboldu, model tekrara girdi). Kırpmak
            # metni sessizce siler; cümle sınırından bölmek içeriği korur.
            budget = self.max_pos_used - 8          # etiketler + </s> payı
            # keep boş LİSTE dönebiliyor (enjeksiyon kapalıyken), None değil.
            # 'keep is None' yazmak bölmeyi hiç tetiklemiyordu.
            if not keep and self._srclen(text, tgt_lang, c, domain) > budget:
                # Parçalara önceki parçanın kuyruğu <ctx> olarak verilecek;
                # o yüzden parçalar CTX_ALLOW kadar daha küçük kesiliyor.
                parts = split_to_budget(
                    text, lambda s: self._srclen(s, tgt_lang, c, domain),
                    budget - CTX_ALLOW)
                self.stats["split"] += 1
            else:
                parts = [text]

            for k, part in enumerate(parts):
                # Parçalar arası SÜREKLİLİK. <ctx> ölçüldü ve çalışıyor:
                #   "The concert was about to begin." bağlamıyla
                #   "She plays beautifully." -> "oynuyor" yerine "çalıyordu"
                # Bölme bu bağlamı koparırsa zamir/çok anlamlılık kayar.
                pctx = c
                if k and len(parts) > 1:
                    tail = self._trim_ctx(parts[k - 1], tgt_lang, domain)
                    pctx = ((list(c) if isinstance(c, (list, tuple)) else [c])
                            if c else []) + [tail]
                encoded.append(self.tok.encode_source(
                    part, tgt_lang, ctx=pctx, domain=domain,
                    max_len=self.max_pos_used,
                    keep_spans=keep if len(parts) == 1 else None))
                todo.append(i)

        self.stats["model"] += len(todo)
        if not todo:
            return out

        # uzunluk kovalama: benzer uzunluklar aynı batch'e -> padding israfı düşer
        pieces = [None] * len(encoded)
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
        for s in range(0, len(order), batch_size):
            chunk = order[s:s + batch_size]
            res = self._beam_batch([encoded[i] for i in chunk], beam,
                                   max_ratio, max_new, len_penalty)
            for i, r in zip(chunk, res):
                pieces[i] = r
        # bölünen satırların parçalarını sırayla birleştir
        joined = {}
        for i, j in enumerate(todo):
            joined.setdefault(j, []).append(pieces[i])
        for j, ps in joined.items():
            r = " ".join(p for p in ps if p)
            out[j] = repair_output(lines[j], r) if self.repair else r
        return out

    def _srclen(self, text, tgt_lang, c, domain):
        return len(self.tok.encode_source(text, tgt_lang, ctx=c, domain=domain))

    def _trim_ctx(self, text, tgt_lang, domain):
        """Önceki parçanın SON cümlelerini CTX_ALLOW token bütçesine sığdır.
        Bağlam baştan değil SONDAN alınır: çeviriyi etkileyen, hemen önceki
        cümledir."""
        sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
        out = ""
        for s in reversed(sents):
            cand = (s + " " + out).strip() if out else s
            if self._srclen(cand, tgt_lang, None, domain) > CTX_ALLOW:
                break
            out = cand
        return out or sents[-1] if sents else text

    # ------------------------------------------------------------------
    def _beam_batch(self, seqs, beam, max_ratio, max_new, len_penalty):
        dev = self.device
        B, K = len(seqs), max(1, beam)
        S = max(len(s) for s in seqs)
        src = torch.full((B, S), PAD_ID, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            src[i, :len(s)] = torch.tensor(s, device=dev)
        src_pad = src != PAD_ID

        mem = self.model.encode(src, src_pad)                    # BİR KEZ
        # her örneği K kez tekrarla -> beam'ler batch boyutunda taşınır
        mem = mem.repeat_interleave(K, dim=0)
        bpad = src_pad.repeat_interleave(K, dim=0)

        cap = max_new or min(self.max_pos_used - 1,
                             int(max_ratio * S) + 8)
        caches = self.model.init_cache(mem, max_len=cap + 1)

        ys = torch.full((B * K, 1), BOS_ID, dtype=torch.long, device=dev)
        scores = torch.full((B, K), float("-inf"), device=dev)
        scores[:, 0] = 0.0                       # başta yalnız 1 beam canlı
        scores = scores.view(-1)
        done = torch.zeros(B * K, dtype=torch.bool, device=dev)

        for t in range(cap):
            # src_pad ZORUNLU: farklı uzunluktaki cümleler aynı batch'te
            # pad'lendiği için cross-attention pad konumlarına da bakar ve
            # kısa cümlelerin çevirisi sessizce bozulur. Tek cümlelik çağrıda
            # fark edilmez, toplu çeviride ortaya çıkar.
            logits = self.model.decode_step(ys[:, -1:], caches, t,
                                            src_pad=bpad)[:, -1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            # bitmiş beam'ler yalnız <pad> üretsin, skorları donsun
            logp[done] = float("-inf")
            logp[done, PAD_ID] = 0.0

            cand = scores.unsqueeze(1) + logp                     # (B*K, V)
            cand = cand.view(B, K * self.cfg.vocab_size)
            top_score, top_idx = cand.topk(K, dim=-1)
            beam_idx = top_idx // self.cfg.vocab_size             # (B, K)
            tok_idx = top_idx % self.cfg.vocab_size

            flat = (torch.arange(B, device=dev).unsqueeze(1) * K + beam_idx).view(-1)
            ys = torch.cat([ys[flat], tok_idx.view(-1, 1)], dim=1)
            scores = top_score.view(-1)
            done = done[flat] | (tok_idx.view(-1) == EOS_ID)
            self._reorder(caches, flat)

            if bool(done.all()):
                break

        ys = ys.view(B, K, -1)
        scores = scores.view(B, K)
        out = []
        for b in range(B):
            best, best_s = None, float("-inf")
            for k in range(K):
                seq = ys[b, k, 1:].tolist()
                if EOS_ID in seq:
                    seq = seq[:seq.index(EOS_ID)]
                s = float(scores[b, k]) / max(1, len(seq)) ** len_penalty
                if s > best_s:
                    best, best_s = seq, s
            out.append(self.tok.decode(best or []))
        return out

    @staticmethod
    def _reorder(caches, idx):
        """Beam yeniden sıralandığında KV cache'leri de aynı sırayla taşı.
        Bu adım atlanırsa beam'ler başkasının geçmişini okur — sessiz bozulma."""
        self_caches, cross_kvs = caches
        for i, (k, v) in enumerate(self_caches):
            self_caches[i] = (k.index_select(0, idx), v.index_select(0, idx))
        for i, (k, v) in enumerate(cross_kvs):
            cross_kvs[i] = (k.index_select(0, idx), v.index_select(0, idx))


# ------------------------------ SRT ------------------------------

def translate_srt(tr, path, tgt_lang, beam):
    """Altyazı dosyası çevir. Zaman kodları ve numaralar korunur, sadece
    metin blokları çevrilir; blok içi satır sonları geri konur."""
    import re
    blocks = open(path, encoding="utf-8-sig").read().split("\n\n")
    texts, slots = [], []
    for bi, b in enumerate(blocks):
        lines = b.strip().split("\n")
        if len(lines) >= 3 and "-->" in lines[1]:
            texts.append(" ".join(lines[2:]))
            slots.append(bi)
    print(f"{len(texts)} altyazı bloğu çevriliyor...", file=sys.stderr)
    t0 = time.time()
    outs = tr.translate(texts, tgt_lang=tgt_lang, domain="sub", beam=beam)
    el = time.time() - t0
    print(f"{el:.1f}s  ({1000*el/max(1,len(texts)):.0f} ms/satır)", file=sys.stderr)
    for bi, o in zip(slots, outs):
        lines = blocks[bi].strip().split("\n")
        blocks[bi] = "\n".join(lines[:2] + [o])
    return "\n\n".join(blocks)


# ------------------------------ SERVER ------------------------------

def serve(tr):
    """Kalıcı servis: stdin'den satır başına bir JSON isteği, stdout'a bir yanıt.

    Boru hattı (pipe) her işletim sisteminde çalışır — soket, D-Bus, portal ya
    da masaüstü servisi gerekmez. Çağıran script modeli hiç bilmez, sadece metin
    gönderir. Model süreç başına BİR KEZ yüklenir (~2 sn), sonrası anlıktır.

      {"lines": ["Hello"], "to": "tr", "domain": "sub", "ctx": ["previous line"]}
      -> {"ok": true, "out": ["Merhaba"], "ms": 23.1}
    """
    print(json.dumps({"ready": True, **tr.info}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            req = json.loads(line)
            out = tr.translate(req.get("lines") or [],
                               tgt_lang=req.get("to", "tr"),
                               ctx=req.get("ctx"),
                               domain=req.get("domain", "sub"),
                               beam=int(req.get("beam", 2)),
                               batch_size=int(req.get("batch", 32)))
            resp = {"ok": True, "out": out, "ms": round(1000 * (time.time() - t0), 1)}
        except Exception as e:                                  # noqa: BLE001
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(resp, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Vexira çeviri")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--spm", default=None,
                    help="yoksa: $VEXIRA_SPM -> ckpt yanı -> havuz")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32-store", action="store_true",
                    help="ağırlığı fp16'ya yuvarlama (biraz daha fazla RAM)")
    ap.add_argument("--to", default="tr", choices=["tr", "en"])
    ap.add_argument("--domain", default="sub", choices=["sub", "ocr", "ui", "doc"])
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--text", nargs="+")
    ap.add_argument("--file")
    ap.add_argument("--srt")
    ap.add_argument("--out")
    ap.add_argument("--server", action="store_true")
    ap.add_argument("--glossary", default=None,
                    help="harici terim TSV'si (verilmezse ckpt'ye gömülü olan)")
    ap.add_argument("--no-glossary", action="store_true",
                    help="sözlüğü kapat — ham model çıktısı (A/B karşılaştırma)")
    ap.add_argument("--inject", action="store_true",
                    help="cümle içi terim enjeksiyonu (ince ayar SONRASI aç; "
                         "ön-eğitilmiş model koruma bölgesine uymuyor)")
    ap.add_argument("--no-repair", action="store_true",
                    help="çıktı onarımını kapat (kaçış dizisi + yer tutucu "
                         "geri yazma). Ölçüldü: bozuk yer tutucu 58 -> 10")
    ap.add_argument("--self-test", action="store_true",
                    help="padding bağımsızlığı regresyon testi (model gerekmez)")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = otomatik (çekirdek sayısı ne olursa olsun en fazla "
                         "DEFAULT_THREADS). Fazla thread bu boyutta YAVAŞLATIR.")
    args = ap.parse_args()

    # Thread'i ckpt'teki ayardan belirle (model yüklenmeden önce gerekli, o
    # yüzden yalnız 'runtime' alanı okunuyor — ağırlıklar değil).
    _rt = {}
    if not args.self_test and os.path.exists(args.ckpt):
        try:
            _rt = torch.load(args.ckpt, map_location="cpu", weights_only=False,
                             mmap=True).get("runtime") or {}
        except Exception:                                       # noqa: BLE001
            _rt = {}
    torch.set_num_threads(resolve_runtime(_rt, threads=args.threads)["threads"])

    if args.self_test:
        return 0 if _test_padding_invariance() else 1

    tr = Translator(args.ckpt, args.spm, args.device,
                    fp16_store=not args.fp32_store,
                    glossary=args.glossary, use_glossary=not args.no_glossary,
                    repair=not args.no_repair, inject=args.inject)
    print(f"[model] {tr.info['params']/1e6:.1f}M param, adım {tr.info['step']}, "
          f"val {tr.info['best_val']}, sözlük {tr.info['glossary']} terim",
          file=sys.stderr)

    if args.server:
        return serve(tr)

    if args.srt:
        res = translate_srt(tr, args.srt, args.to, args.beam)
        (open(args.out, "w", encoding="utf-8") if args.out else sys.stdout).write(res)
        return 0

    if args.file:
        # '#' ile başlayan satırlar yorum sayılır ve çeviriye girmez —
        # samples/ altındaki örnek dosyalar bölüm başlıklarını böyle işaretliyor.
        lines = [l.rstrip("\n") for l in open(args.file, encoding="utf-8")
                 if l.strip() and not l.lstrip().startswith("#")]
    elif args.text:
        lines = args.text
    else:
        lines = [l.rstrip("\n") for l in sys.stdin if l.strip()]

    t0 = time.time()
    outs = tr.translate(lines, tgt_lang=args.to, domain=args.domain, beam=args.beam)
    el = time.time() - t0
    sink = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    for o in outs:
        sink.write(o + "\n")
    print(f"[hız] {len(lines)} satır, {el:.2f}s, "
          f"{1000*el/max(1,len(lines)):.1f} ms/satır", file=sys.stderr)
    return 0




# =========================== REGRESYON TESTİ ===========================

def _test_padding_invariance():
    """Toplu çeviride kaynak padding'i çıktıyı ETKİLEMEMELİ.

    Yakalanan hata: _beam_batch, src_pad'i decode_step'e geçirmiyordu ->
    cross-attention encoder memory'sindeki pad konumlarına da bakıyordu.
    Tek cümle çevirirken fark edilmez (pad yok); farklı uzunlukta cümleler
    aynı batch'e girince kısa olanların çevirisi sessizce bozulur.

    Karşılaştırma LOGIT seviyesinde yapılıyor. Üretilen token'ları karşılaştırmak
    işe yaramıyor: rastgele başlatılmış modelde argmax girdiden bağımsız olarak
    hep aynı token'a düşüyor, test boşuna geçiyor.

    python translate.py --self-test
    """
    import torch as _t
    from config import get_config
    from model import Vexira

    _t.manual_seed(0)
    cfg = get_config("tiny", dropout=0.0)
    model = Vexira(cfg).eval()

    short = _t.tensor([[4, 100, 101, 3]])
    S = 42
    long_ = _t.tensor([[4] + list(range(200, 240)) + [3]])
    padded = _t.full((2, S), PAD_ID, dtype=_t.long)
    padded[0, :4] = short[0]
    padded[1] = long_[0]
    pad_mask = padded != PAD_ID

    ys = _t.tensor([[BOS_ID]])

    with _t.no_grad():
        # (a) kısa cümle TEK BAŞINA (padding yok) — referans
        mem_a = model.encode(short, short != PAD_ID)
        ca = model.init_cache(mem_a, max_len=4)
        ref = model.decode_step(ys, ca, 0, src_pad=(short != PAD_ID))[0, -1]

        # (b) pad'li batch, MASKE İLE -> referansla aynı olmalı
        mem_b = model.encode(padded, pad_mask)
        cb = model.init_cache(mem_b, max_len=4)
        with_mask = model.decode_step(
            _t.tensor([[BOS_ID], [BOS_ID]]), cb, 0, src_pad=pad_mask)[0, -1]

        # (c) pad'li batch, MASKESİZ -> referanstan FARKLI olmalı (hata bu)
        cc = model.init_cache(mem_b, max_len=4)
        no_mask = model.decode_step(
            _t.tensor([[BOS_ID], [BOS_ID]]), cc, 0, src_pad=None)[0, -1]

    d_mask = float((ref - with_mask).abs().max())
    d_none = float((ref - no_mask).abs().max())

    ok = True
    c1 = d_mask < 1e-4
    ok &= c1
    print(f"  {'✓' if c1 else '✗'} maskeli batch == tek cümle   (maxdiff {d_mask:.2e})")
    c2 = d_none > 1e-3
    ok &= c2
    print(f"  {'✓' if c2 else '✗'} maskesiz batch != tek cümle  (maxdiff {d_none:.2e})"
          + ("" if c2 else "  <- test anlamsız, fark üretmiyor"))

    # uçtan uca: _beam_batch gerçekten maskeyi geçiriyor mu
    src_seen = {}
    orig = model.decode_step

    def _spy(tgt_ids, caches, pos, src_pad=None):
        src_seen["got"] = src_pad is not None
        return orig(tgt_ids, caches, pos, src_pad=src_pad)

    model.decode_step = _spy

    class _IdTok:
        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    class _Stub:
        pass
    tr = _Stub()
    tr.device = _t.device("cpu")
    tr.cfg = cfg
    tr.model = model
    tr.tok = _IdTok()
    tr._beam_batch = Translator._beam_batch.__get__(tr)
    tr._reorder = Translator._reorder
    tr._beam_batch([[4, 100, 101, 3], [4] + list(range(200, 240)) + [3]], 2, 2.0, 6, 1.0)
    model.decode_step = orig

    c3 = src_seen.get("got", False)
    ok &= c3
    print(f"  {'✓' if c3 else '✗'} _beam_batch src_pad'i decode_step'e geçiriyor")

    print("\n" + ("PADDING BAĞIMSIZLIĞI ✓" if ok else "BAŞARISIZ ✗"))
    return ok


if __name__ == "__main__":
    sys.exit(main())
