# -*- coding: utf-8 -*-
"""
Temiz bitext -> uint16 token .bin + index .npy (eğitim formatı).

Her çift İKİ YÖNDE üretilir:
    <2tr> [bağlam <ctx>] İngilizce metin </s>   ->   Türkçe metin </s>
    <2en> [bağlam <ctx>] Türkçe metin   </s>   ->   İngilizce metin </s>
Böylece tek model çift yönlü çalışır ve veri 2x olur.

BAĞLAM (<ctx>): aynı doc_id içindeki önceki 1-2 satır kaynak tarafın başına
eklenir. "bank -> kıyı mı banka mı", "play -> oynamak mı çalmak mı" ayrımı
burada çözülür. Çiftlerin yalnızca --ctx-prob kadarında uygulanır ki model
bağlamsız da (canlı OCR'da ilk satır) çalışabilsin.

Kayıp yalnızca hedef üzerinden hesaplanır; bağlam kaynak tarafta olduğu için
zaten kayba girmez.

Çıktı:
  vexira.bin        uint16 düz token dizisi
  vexira.idx.npy    int64 (N,4) = [src_off, src_len, tgt_off, tgt_len]
  vexira.meta.json  {vocab, n_examples, n_tokens, directions, spm_sha, ...}

Kullanım:
  python build_bin.py
  python build_bin.py --max-len 128 --ctx-prob 0.5
  python build_bin.py --eval-only     # FLORES devtest -> ayrı bin
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from collections import Counter

from array import array

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import VexiraTokenizer  # noqa: E402

POOL = os.environ.get("VEXIRA_POOL", os.path.expanduser("~/ai-data/vexira"))
CLEAN_DIR = os.path.join(POOL, "clean")
EVAL_DIR = os.path.join(POOL, "eval")
TOK_DIR = os.path.join(POOL, "tokenized")
SPM_PATH = os.path.join(POOL, "spm", "vexira_spm.model")

# altyazı kaynakları -> <sub>, yazılım -> <ui>, geri kalanı <doc>
DOMAIN_OF = {
    # <sub>: konuşma dili / altyazı satırı
    "opensubs": "sub", "qed": "sub", "ted2020": "sub",
    "neulab_ted": "sub", "ted2013": "sub",
    # <ui>: yazılım arayüz metni — kısa, emirli, placeholder'lı
    "kde4": "ui", "ubuntu": "ui", "php": "ui",
    # <doc>: düzyazı / ansiklopedik / resmi
    "setimes": "doc", "wikimatrix": "doc", "wikimedia": "doc",
    "wikipedia": "doc", "wmtnews": "doc", "globalvoices": "doc",
    "eubookshop": "doc", "infopankki": "doc", "bible": "doc", "tanzil": "doc",
    "hplt": "doc", "ccmatrix": "doc", "multicc": "doc", "ccaligned": "doc",
    "nllb": "doc",
    # etiketsiz: tek cümle/varlık, domain sinyali yanıltıcı olur
    "tatoeba": None, "xlent": None, "wikititles": None,
}


class BinWriter:
    """Token akışını .bin'e, indeksi ayrı ham dosyaya AKITARAK yazar.

    İndeksi Python listesinde biriktirmek 100M+ örnekte belleği patlatır:
    satır başına tuple (72 B) + iki büyük int nesnesi (~64 B) + liste işaretçisi
    = ~144 B; 120M örnek -> ~17 GB. Bunun yerine array('q') tamponu ham int64
    olarak diske akıtılıyor (8 B/değer), bellek sabit kalıyor.
    """

    FLUSH_ROWS = 200_000

    def __init__(self, prefix):
        self.prefix = prefix
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        self.f = open(prefix + ".bin.tmp", "wb")
        self.fidx = open(prefix + ".idx.tmp", "wb")
        self.idxbuf = array("q")          # düz int64: off, src_len, tgt_off, tgt_len
        self.n = 0
        self.off = 0
        self.buf = []

    def add(self, src_ids, tgt_ids):
        sl, tl = len(src_ids), len(tgt_ids)
        self.idxbuf.extend((self.off, sl, self.off + sl, tl))
        self.buf.append(np.asarray(src_ids + tgt_ids, dtype=np.uint16))
        self.off += sl + tl
        self.n += 1
        if len(self.buf) >= 100_000:
            self._flush()
        if len(self.idxbuf) >= self.FLUSH_ROWS * 4:
            self._flush_idx()

    def _flush(self):
        if self.buf:
            np.concatenate(self.buf).tofile(self.f)
            self.buf = []

    def _flush_idx(self):
        if len(self.idxbuf):
            self.idxbuf.tofile(self.fidx)
            self.idxbuf = array("q")

    def close(self, meta):
        self._flush()
        self._flush_idx()
        self.f.close()
        self.fidx.close()
        os.replace(self.prefix + ".bin.tmp", self.prefix + ".bin")
        # ham int64 -> (N,4) .npy  (tek numpy dizisi, Python nesnesi yok)
        idx = np.fromfile(self.prefix + ".idx.tmp", dtype=np.int64).reshape(-1, 4)
        np.save(self.prefix + ".idx.npy", idx)
        del idx
        os.remove(self.prefix + ".idx.tmp")
        meta = dict(meta, n_examples=self.n, n_tokens=int(self.off))
        json.dump(meta, open(self.prefix + ".meta.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        return self.n, self.off

    @property
    def index(self):                       # geriye dönük uyum (limit kontrolü)
        return range(self.n)


def iter_rows(files):
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def add_replay(w, prefix, n, seed, max_len, min_tgt=0):
    """Ön-eğitim bin'inden n örneği OLDUĞU GİBİ kopyala (yeniden tokenize yok).

    İki işi birden yapar:

    1) UNUTMAYI ÖNLER. İnce ayar seti tek domain'e yığıldığında (ölçüldü:
       %98 <ui>) model o domain'e kayar ve altyazı/düzyazı çevirisi bozulur.

    2) UZUNLUK DAĞILIMINI DÜZELTİR (min_tgt ile). İnce ayar seti kısaya yığılı;
       ön-eğitim korpusunda hedefi 30+ token olan örnek bol (%18). Bunlar
       İNSAN HİZALI gerçek bitext — sentetik uzun metin üretmekten hem daha
       kaliteli hem bedava. Sentetik kitap metninde ölçülen hatalar:
       "Ben adım Elif", "kahkalar", "kalınduvarlı" — modele bunları öğretmek
       uzunluk kazancından pahalıya gelir.
    """
    idx_path, bin_path = prefix + ".idx.npy", prefix + ".bin"
    if not (os.path.exists(idx_path) and os.path.exists(bin_path)):
        print(f"  ! tekrar atlandı, ön-eğitim seti yok: {prefix}")
        return 0
    idx = np.load(idx_path, mmap_mode="r")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)

    if min_tgt:
        # Uzun örnekleri bulmak için tüm indeksi taramak yerine bol örnekleme
        # yapılıp süzülüyor: %18'i eşiği geçtiği için 8x aday fazlasıyla yeter.
        cand = rng.choice(len(idx), size=min(len(idx), n * 8), replace=False)
        cand = np.sort(cand)
        keep = [i for i in cand if idx[i][3] >= min_tgt][:n]
        pick = np.array(keep, dtype=np.int64)
        print(f"  tekrar: hedefi >= {min_tgt} token olanlardan seçiliyor "
              f"({len(cand):,} aday -> {len(pick):,})")
    else:
        pick = np.sort(rng.choice(len(idx), size=min(n, len(idx)), replace=False))

    added = 0
    for i in pick:                             # sıralı okuma = memmap dostu
        so, sl, to, tl = (int(x) for x in idx[i])
        if sl > max_len or tl > max_len:
            continue
        w.add(data[so:so + sl].tolist(), data[to:to + tl].tolist())
        added += 1
    print(f"  tekrar: ön-eğitimden {added:,} örnek karıştırıldı")
    return added


def build(files, out_prefix, tok, max_len, ctx_prob, ctx_lines, seed,
          use_domain=True, limit=0, row_domain=False,
          replay=0, replay_from=None, replay_min_tgt=0, replay_long=0):
    rng = random.Random(seed)
    w = BinWriter(out_prefix)
    dirs = Counter()
    dropped = Counter()
    doms = Counter()
    n_ctx = 0

    prev_doc = None
    hist = []            # aynı doc_id içindeki önceki (en, tr) satırlar

    for k, r in enumerate(iter_rows(files)):
        if limit and w.index and len(w.index) >= limit:
            break
        en, tr = r.get("src", ""), r.get("tgt", "")
        if not en or not tr:
            continue
        doc = r.get("doc_id")
        if doc != prev_doc:
            prev_doc, hist = doc, []

        src_tag = r.get("source")
        if row_domain:
            # ince ayar: domain satırın kendisinde yazar (kaynak adına bakma).
            # Üretilen SFT verisinde tek dosyada <ui>/<sub>/<doc> karışık olur.
            domain = r.get("domain") or None
        else:
            domain = DOMAIN_OF.get(src_tag) if use_domain else None
        doms[domain or "-"] += 1

        # bağlam yalnız doc_id BİLİNİYORSA anlamlı; yoksa ardışıklık garantisi yok
        use_ctx = bool(doc) and hist and rng.random() < ctx_prob
        ctx_en = [h[0] for h in hist[-ctx_lines:]] if use_ctx else None
        ctx_tr = [h[1] for h in hist[-ctx_lines:]] if use_ctx else None
        if use_ctx:
            n_ctx += 1

        # SFT'de bir örnek tek yöne kilitlenebilir ("dir") ve kaynak tarafta
        # sözlük enjeksiyonu taşıyabilir ("keep": [[b,e],...] karakter aralığı).
        # Enjeksiyon biçimini eğitimde GÖRMEZSE model çıkarımda o biçime uymaz —
        # <keep_start>..<keep_end> yalnız placeholder için öğrenilmiş olur.
        only = r.get("dir") if row_domain else None
        keep = r.get("keep") if row_domain else None

        for tgt_lang, s_text, t_text, s_ctx in (
                ("tr", en, tr, ctx_en),          # <2tr> EN -> TR
                ("en", tr, en, ctx_tr)):         # <2en> TR -> EN
            if only and only != ("en2tr" if tgt_lang == "tr" else "tr2en"):
                continue
            s_ids = tok.encode_source(s_text, tgt_lang, ctx=s_ctx,
                                      domain=domain, max_len=max_len,
                                      keep_spans=keep if only else None)
            t_ids = tok.encode_target(t_text, max_len=max_len)
            if len(t_ids) < 2 or len(s_ids) < 3:
                dropped["kısa"] += 1
                continue
            if len(s_ids) > max_len or len(t_ids) > max_len:
                dropped["uzun"] += 1
                continue
            w.add(s_ids, t_ids)
            dirs[f"{'en' if tgt_lang == 'tr' else 'tr'}->{tgt_lang}"] += 1

        hist.append((en, tr))
        if len(hist) > ctx_lines:
            hist.pop(0)

        if (k + 1) % 1_000_000 == 0:
            print(f"    {(k+1)/1e6:.1f}M çift işlendi -> {len(w.index):,} örnek",
                  flush=True)

    # İKİ AYRI TEKRAR HAVUZU:
    #   düz  -> domain dengesi (unutmayı önler), ön-eğitim dağılımını taşır
    #   uzun -> yalnız uzunluk açığını kapatır
    # Tek havuzla denendi ve ölçüldü: hepsi düz olunca >=30 token %12.9 (hedef
    # %18.1'in altında), hepsi uzun olunca %36.2 (iki katı aşıyor). Karışım şart.
    n_replay = add_replay(w, replay_from, replay, seed, max_len) if replay else 0
    if replay_long:
        n_replay += add_replay(w, replay_from, replay_long, seed + 1, max_len,
                               min_tgt=replay_min_tgt or 30)
    if n_replay:
        doms["<tekrar>"] = n_replay
    sha = hashlib.sha256(open(tok.model_path, "rb").read()).hexdigest()[:16]
    n, ntok = w.close({"vocab": tok.vocab_size, "spm_sha": sha,
                       "max_len": max_len, "ctx_prob": ctx_prob,
                       "ctx_lines": ctx_lines,
                       "directions": dict(dirs), "dropped": dict(dropped),
                       "domains": dict(doms), "with_context": n_ctx})
    return n, ntok, dirs, dropped, n_ctx, doms


def main():
    ap = argparse.ArgumentParser(description="Vexira token .bin derleyici")
    ap.add_argument("--clean", default=CLEAN_DIR)
    ap.add_argument("--out", default=os.path.join(TOK_DIR, "vexira"))
    ap.add_argument("--spm", default=SPM_PATH)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--ctx-prob", type=float, default=0.5,
                    help="bağlam eklenen çift oranı (0 = bağlam kapalı)")
    ap.add_argument("--ctx-lines", type=int, default=2)
    ap.add_argument("--no-domain", action="store_true", help="<sub>/<ui> etiketi ekleme")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="sadece FLORES devtest -> vexira_flores.bin")
    ap.add_argument("--finetune", action="store_true",
                    help="ince ayar (SFT) seti derle: domain SATIRDAN okunur, "
                         "bağlam kapalı, çıktı vexira_sft.*")
    ap.add_argument("--sft-dir", default=os.path.join(POOL, "finetune"),
                    help="--finetune ile okunacak jsonl klasörü")
    ap.add_argument("--replay", type=int, default=0,
                    help="ön-eğitim setinden bu kadar örnek KARIŞTIR. İnce ayar "
                         "seti tek domain'e yığıldığında (ör. %%98 <ui>) model "
                         "genel çeviriyi unutur; tekrar (replay) bunu önler.")
    ap.add_argument("--replay-long", type=int, default=0,
                    help="AYRICA bu kadar UZUN örnek karıştır (hedef >= "
                         "--replay-min-tgt token). Uzunluk açığını kapatır.")
    ap.add_argument("--replay-min-tgt", type=int, default=30,
                    help="tekrar örneklerini hedefi >= N token olanlardan seç. "
                         "Uzunluk dağılımını sentetik veri üretmeden düzeltir.")
    ap.add_argument("--replay-from", default=os.path.join(TOK_DIR, "vexira"),
                    help="tekrar örneklerinin alınacağı ön-eğitim prefix'i")
    args = ap.parse_args()

    tok = VexiraTokenizer(args.spm)
    print(f"tokenizer: {args.spm}  vocab={tok.vocab_size}")

    if args.eval_only:
        f = os.path.join(EVAL_DIR, "flores200_devtest.jsonl")
        if not os.path.exists(f):
            print(f"yok: {f}\nönce: python ~/ai-data/vexira/pipeline/fetch_bitext.py --flores")
            return 1
        out = os.path.join(TOK_DIR, "vexira_flores")
        n, ntok, dirs, dropped, nctx, doms = build(
            [f], out, tok, args.max_len, 0.0, 0, args.seed, use_domain=False)
        print(f"-> {out}.bin  {n:,} örnek, {ntok/1e6:.2f}M token, {dict(dirs)}")
        return 0

    if args.finetune:
        files = sorted(glob.glob(os.path.join(args.sft_dir, "*.jsonl")))
        if not files:
            print(f"ince ayar verisi yok: {args.sft_dir}\n"
                  f"önce: python ~/ai-data/vexira/pipeline/gen_finetune.py term|dialog|long")
            return 1
        out = args.out if args.out != os.path.join(TOK_DIR, "vexira") \
            else os.path.join(TOK_DIR, "vexira_sft")
        print(f"{len(files)} SFT dosyası -> {out}  (bağlam KAPALI, domain satırdan)")
        # ctx_prob=0: SFT örnekleri bağımsız cümleler, doc_id yok.
        n, ntok, dirs, dropped, nctx, doms = build(
            files, out, tok, args.max_len, 0.0, 0, args.seed,
            row_domain=True, limit=args.limit,
            replay=args.replay, replay_from=args.replay_from,
            replay_min_tgt=args.replay_min_tgt, replay_long=args.replay_long)
        print(f"\n{n:,} örnek, {ntok/1e6:.2f}M token")
        for d, c in dirs.most_common():
            print(f"  {d:10s} {c:>10,}")
        print("  domain:", dict(doms))
        if dropped:
            print("  düşen:", dict(dropped))
        print(f"-> {out}.bin / .idx.npy / .meta.json")
        return 0

    files = sorted(glob.glob(os.path.join(args.clean, "*.jsonl")))
    if not files:
        print(f"temiz veri yok: {args.clean}\nönce: python ~/ai-data/vexira/pipeline/clean_bitext.py")
        return 1
    print(f"{len(files)} temiz dosya, max_len={args.max_len}, "
          f"ctx_prob={args.ctx_prob}, ctx_lines={args.ctx_lines}")

    n, ntok, dirs, dropped, nctx, doms = build(
        files, args.out, tok, args.max_len, args.ctx_prob, args.ctx_lines,
        args.seed, use_domain=not args.no_domain, limit=args.limit)

    print(f"\n{'='*58}")
    print(f"{n:,} eğitim örneği, {ntok/1e6:.1f}M token "
          f"({ntok*2/1e9:.2f} GB uint16)")
    for d, c in dirs.most_common():
        print(f"  {d:10s} {c:>12,}")
    print(f"  bağlamlı  {nctx:>12,} kaynak çift")
    print("  domain:", dict(doms))
    if dropped:
        print("  düşen:", dict(dropped))
    print(f"-> {args.out}.bin / .idx.npy / .meta.json")
    print(f"\nsıradaki:\n  python build_bin.py --eval-only")
    print(f"  python train_local.py --data {args.out} --preset tiny --max-steps 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
