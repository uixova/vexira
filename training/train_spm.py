# -*- coding: utf-8 -*-
"""
Vexira ortak TR+EN SentencePiece unigram tokenizer eğitimi.

model_type="unigram" (BPE DEĞİL): Türkçe sondan eklemeli. Unigram, olasılıksal
budama ile ekleri ("-lar", "-ımız", "-dan") kendi başına parça olarak tutmaya
BPE'nin açgözlü birleştirmesinden daha yatkın. Tek ek yüzlerce kelimeye
yayıldığı için devasa sözlüğe gerek kalmıyor — 32k iki dile fazlasıyla yetiyor.

normalization_rule_name="nmt_nfkc": NFKC normalizasyonu var, KÜÇÜK HARFE İNDİRME
YOK. Küçültme yapan bir tokenizer çeviri çıktısında özel isim ve cümle başı
büyük harfini kalıcı olarak kaybettirir — bu iş için ölümcül.

byte_fallback=True + character_coverage=1.0: <unk> imkansız. Emoji, nadir
karakter, bozuk kodlama — hepsi bayta düşerek temsil edilir.

⚠️ ÖZEL TOKEN SIRASI DONDURULMUŞTUR. tokenizer.py:USER_SYMBOLS listesinin
ortasına ekleme yapmak tüm id'leri kaydırır = eğitilmiş modelin tamamı çöp.
Yeni ihtiyaç <reserved_N> slotlarından karşılanır.

Kullanım:
  python train_spm.py                 # clean/ üzerinden eğit
  python train_spm.py --vocab 40000   # TR fertility yüksekse
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import USER_SYMBOLS, PAD_ID, UNK_ID, BOS_ID, EOS_ID  # noqa: E402

POOL = os.environ.get("VEXIRA_POOL", os.path.expanduser("~/ai-data/vexira"))
CLEAN_DIR = os.path.join(POOL, "clean")
SPM_DIR = os.path.join(POOL, "spm")


def build_sample(clean_dir, out_path, per_lang, seed=0):
    """Her dilden dengeli örnek çıkar. Dengesiz olursa vocab baskın dile kayar
    ve diğer dilin fertility'si patlar."""
    files = sorted(glob.glob(os.path.join(clean_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"temiz veri yok: {clean_dir}\n"
                                f"önce: python ~/ai-data/vexira/pipeline/clean_bitext.py")
    rng = random.Random(seed)
    # İki geçiş yerine rezervuar benzeri: her satırı p olasılıkla al. Toplam
    # satır sayısını bilmediğimiz için önce kabaca say.
    total = 0
    for p in files:
        with open(p, "rb") as f:
            total += sum(1 for _ in f)
    keep_p = min(1.0, per_lang / max(1, total))
    print(f"  {total:,} temiz çift, örnekleme oranı {keep_p:.4f} "
          f"(hedef dil başına {per_lang:,})")

    n_en = n_tr = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in files:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if rng.random() > keep_p:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    out.write(r["src"].replace("\n", " ") + "\n")
                    out.write(r["tgt"].replace("\n", " ") + "\n")
                    n_en += 1
                    n_tr += 1
    print(f"  örnek: {n_en:,} EN + {n_tr:,} TR satır -> {out_path}")
    return n_en + n_tr


def main():
    ap = argparse.ArgumentParser(description="Vexira SentencePiece eğitimi")
    ap.add_argument("--clean", default=CLEAN_DIR)
    ap.add_argument("--out-dir", default=SPM_DIR)
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--per-lang", type=int, default=2_000_000,
                    help="dil başına örnek satır")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import sentencepiece as spm

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, "vexira_spm")
    if os.path.exists(prefix + ".model") and not args.force:
        print(f"zaten var: {prefix}.model  (--force ile yeniden eğit)")
        print("⚠️  Yeniden eğitmek TÜM token .bin'lerini ve eğitilmiş modeli "
              "geçersiz kılar — vocab kayması resume'u öldürür.")
        return 0

    sample = os.path.join(args.out_dir, "_spm_sample.txt")
    print("[1/3] örnek metin çıkarılıyor")
    build_sample(args.clean, sample, args.per_lang, args.seed)

    print(f"[2/3] unigram eğitimi (vocab={args.vocab}, {args.threads} thread)")
    spm.SentencePieceTrainer.train(
        input=sample,
        model_prefix=prefix,
        model_type="unigram",
        vocab_size=args.vocab,
        character_coverage=1.0,          # byte_fallback ile birlikte <unk> imkansız
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",   # LOWERCASE YOK
        split_digits=True,               # "1990" -> 1 9 9 0, sayı ezberi olmasın
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,  # UI metninde boşluk anlamlı olabilir
        user_defined_symbols=USER_SYMBOLS,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        num_threads=args.threads,
        input_sentence_size=8_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=False,
    )

    print("[3/3] doğrulama")
    sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
    ok = True

    for name, got, want in (("pad", sp.pad_id(), PAD_ID), ("unk", sp.unk_id(), UNK_ID),
                            ("bos", sp.bos_id(), BOS_ID), ("eos", sp.eos_id(), EOS_ID)):
        good = got == want
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}_id = {got} (beklenen {want})")

    missing = [s for s in USER_SYMBOLS if sp.piece_to_id(s) == sp.unk_id()]
    ok &= not missing
    print(f"  {'✓' if not missing else '✗'} {len(USER_SYMBOLS)} özel token sözlükte"
          + (f" — EKSİK: {missing}" if missing else ""))

    # özel tokenlar TEK parça kalmalı, alt parçalara bölünmemeli
    split = [s for s in USER_SYMBOLS if len(sp.encode(s, out_type=int)) != 1]
    ok &= not split
    print(f"  {'✓' if not split else '✗'} özel tokenlar tek parça"
          + (f" — BÖLÜNEN: {split}" if split else ""))

    # round-trip
    tests = ["Merhaba dünya! Nasılsın?", "The quick brown fox jumps over 42 dogs.",
             "İstanbul'da yaşıyorum, İzmir'e gidiyorum.", "Ünlü şarkıcı çığlık attı."]
    bad = [t for t in tests if sp.decode(sp.encode(t, out_type=int)) != t]
    ok &= not bad
    print(f"  {'✓' if not bad else '✗'} round-trip"
          + (f" — BOZULAN: {bad}" if bad else ""))

    # fertility (token/kelime)
    en_lines, tr_lines = [], []
    with open(sample, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            (en_lines if i % 2 == 0 else tr_lines).append(ln.strip())
            if i > 40000:
                break

    def fert(lines):
        t = sum(len(sp.encode(x, out_type=int)) for x in lines)
        w = sum(max(1, len(x.split())) for x in lines)
        return t / w

    f_en, f_tr = fert(en_lines), fert(tr_lines)
    tok_en = sum(len(sp.encode(x, out_type=int)) for x in en_lines) / max(1, len(en_lines))
    tok_tr = sum(len(sp.encode(x, out_type=int)) for x in tr_lines) / max(1, len(tr_lines))
    print(f"  eğitim korpusu:  EN fert={f_en:.2f} ({tok_en:.1f} tok/satır)   "
          f"TR fert={f_tr:.2f} ({tok_tr:.1f} tok/satır)")

    # DİKKAT: fertility metin türüne çok duyarlı. Altyazı ve web taraması kısa,
    # gayri-resmi, noktalama yoğun -> kelime başına token sayısı doğal olarak
    # yüksek çıkar. Vocab kararı DÜZYAZI üzerinden verilmeli, yoksa gereksiz
    # yere vocab büyütülür (embed şişer, parametre bütçesi katmanlardan çalınır).
    flores = os.path.join(POOL, "eval", "flores200_devtest.jsonl")
    if os.path.exists(flores):
        rows = [json.loads(l) for l in open(flores, encoding="utf-8")]
        pe, pt = fert([r["src"] for r in rows]), fert([r["tgt"] for r in rows])
        print(f"  düzyazı (FLORES): EN fert={pe:.2f} (hedef 1.2-1.5)   "
              f"TR fert={pt:.2f} (hedef 1.5-1.9)")
        if pt > 2.0 or pe > 1.7:
            print("  ⚠️  Düzyazıda bile yüksek — --vocab 40000 ile yeniden eğitmeyi düşün")
        else:
            print("  ✓ vocab boyutu yeterli")
    else:
        print("  (FLORES yok; düzyazı kontrolü atlandı — fetch_bitext.py --flores)")

    sha = hashlib.sha256(open(prefix + ".model", "rb").read()).hexdigest()[:16]
    json.dump({"vocab": sp.get_piece_size(), "sha": sha, "model_type": "unigram",
               "user_symbols": USER_SYMBOLS,
               "fertility": {"en": round(f_en, 3), "tr": round(f_tr, 3)}},
              open(prefix + ".info.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    os.remove(sample)
    print(f"\n{'✓ HAZIR' if ok else '✗ SORUNLU'}: {prefix}.model  "
          f"vocab={sp.get_piece_size()}  sha={sha}")
    if ok:
        print("sıradaki: python build_bin.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
