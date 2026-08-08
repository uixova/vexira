# -*- coding: utf-8 -*-
"""
Vexira tokenizer — ortak TR+EN SentencePiece unigram sarmalayıcısı.

Neden unigram (BPE değil): Türkçe sondan eklemeli. Unigram, ekleri ("-lar",
"-ımız", "-dan") kendi başına parça olarak öğrenmeye BPE'den daha yatkın; tek ek
yüzlerce kelimeye yayıldığı için devasa sözlüğe gerek kalmaz.

Neden bu havuzdaki mevcut BPE vocab'ı DEĞİL: o tokenizer normalize()'de her
şeyi küçük harfe indiriyor -> çeviri çıktısında özel isim ve cümle başı büyük
harfi kaybolur, kabul edilemez. Ayrıca TR-only ve dokunulmaz (başka modellerin
token cache'i ona bağlı, vocab kayması onların resume'unu öldürür).

Dizi formatı:
  kaynak: <2xx> [bağlam <ctx>] [<domain>] metin </s>
  hedef : metin </s>          (decoder girdisi = <s> + hedef[:-1])
"""

import os
import re

# --- kontrol token'ları (SentencePiece'e sabitlenen id'ler) ---
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

LANG_TAGS = ["<2tr>", "<2en>"]
CTX_TAG = "<ctx>"
DOMAIN_TAGS = ["<sub>", "<ocr>", "<ui>", "<doc>"]
AUDIO_TAGS = ["<stt>", "<tts>"]
NEWLINE_TAG = "<nl>"
KEEP_START, KEEP_END = "<keep_start>", "<keep_end>"
RESERVED = [f"<reserved_{i}>" for i in range(8)]

# SIRA ÖNEMLİ — id ataması buna bağlı. Sonradan araya token eklemek TÜM
# token'ların id'sini kaydırır = tam yeniden eğitim. Yeni ihtiyaç çıkarsa
# <reserved_N> kullanılacak, liste ortasına ekleme YAPILMAYACAK.
USER_SYMBOLS = (LANG_TAGS + [CTX_TAG] + DOMAIN_TAGS + AUDIO_TAGS
                + [NEWLINE_TAG, KEEP_START, KEEP_END] + RESERVED)

DEFAULT_MODEL = os.path.expanduser("~/ai-data/vexira/spm/vexira_spm.model")

# Çevrilmemesi gereken kalıplar: oyun/UI placeholder'ları.
#   {player_name}  {0}  %s  %1$d  <color=#fff>  [b]  \n  $VAR
_PLACEHOLDER_RE = re.compile(
    r"(\{[^{}\s]{0,40}\}"          # {0}, {player_name}
    r"|%\d*\$?[sdifux]"            # %s, %d, %1$s
    r"|</?[A-Za-z][^<>]{0,60}>"    # <color=#fff>, </b>
    r"|\[/?[A-Za-z][^\[\]]{0,30}\]"# [b], [/color]
    r"|\$[A-Za-z_][A-Za-z0-9_]*"   # $VAR
    r"|\\[nrt])")                  # \n, \t


def _c2b(text, ci):
    """karakter indeksi -> bayt indeksi (SentencePiece proto ofsetleri bayt)."""
    return len(text[:ci].encode("utf-8"))


class VexiraTokenizer:
    def __init__(self, model_path=DEFAULT_MODEL):
        import sentencepiece as spm
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"tokenizer bulunamadı: {model_path}\n"
                f"önce: python train_spm.py")
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # id'leri VARSAYMA, modelden oku. Sürüm/parametre değişimi id kaydırabilir.
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        for name, got, want in (("pad", self.pad_id, PAD_ID), ("unk", self.unk_id, UNK_ID),
                                ("bos", self.bos_id, BOS_ID), ("eos", self.eos_id, EOS_ID)):
            if got != want:
                raise ValueError(f"{name}_id={got}, beklenen {want} — spm yeniden eğitilmeli")

        self.sym = {s: self.sp.piece_to_id(s) for s in USER_SYMBOLS}
        missing = [s for s, i in self.sym.items() if i == self.unk_id]
        if missing:
            raise ValueError(f"özel tokenlar sözlükte yok: {missing} — spm yeniden eğitilmeli")

    def __len__(self):
        return self.sp.get_piece_size()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def lang_id(self, lang):
        tag = f"<2{lang}>"
        if tag not in self.sym:
            raise KeyError(f"desteklenmeyen hedef dil: {lang} ({LANG_TAGS})")
        return self.sym[tag]

    # ----------------------------- ENCODE -----------------------------
    def _encode_protected(self, text, extra_spans=None):
        """Placeholder'ları <keep_start>…<keep_end> arasına alarak encode et.
        Model bu bölgeyi 'aynen taşı' olarak öğrenir, çevirmeye kalkışmaz.

        Metin TEK SEFERDE encode edilir; işaretler token SINIRLARINA konur.
        Metni parçalara bölüp ayrı ayrı encode etmek olmaz: SentencePiece her
        parçanın başına dummy boşluk ekler, decode'da fazladan boşluk çıkar
        ("Hello {x}, ok" -> "Hello  {x} , ok"). Karakter offsetleri proto
        API'sinden alınıp örtüşen token'lar işaretleniyor.

        extra_spans: sözlükten gelen (başlangıç, bitiş) karakter aralıkları.
        Aynı mekanizma kullanılır — terim karşılığı metne zaten yerleştirilmiş
        olur, burada yalnızca 'dokunma' işareti konur.

        ⚠️ SentencePiece proto'daki pc.begin/pc.end BAYT ofsetidir, karakter
        değil. Regex ve dilim aralıkları ise karakter verir. ASCII'de ikisi
        çakıştığı için hata görünmez; "Türkçe {x}" gibi bir metinde işaret
        yanlış token'a düşer ve koruma bölgesi kayar. Aralıklar karşılaştırmadan
        ÖNCE bayta çevriliyor.
        """
        spans = [(m.start(), m.end()) for m in _PLACEHOLDER_RE.finditer(text)]
        if extra_spans:
            spans += [(int(b), int(e)) for b, e in extra_spans]
        proto = self.sp.encode(text, out_type="proto")
        if not spans:
            return [p.id for p in proto.pieces]
        if not text.isascii():
            spans = [(_c2b(text, b), _c2b(text, e)) for b, e in spans]
        spans.sort()

        out, inside = [], False
        for pc in proto.pieces:
            hit = any(pc.begin < e and pc.end > b for b, e in spans)
            if hit and not inside:
                out.append(self.sym[KEEP_START])
                inside = True
            elif not hit and inside:
                out.append(self.sym[KEEP_END])
                inside = False
            out.append(pc.id)
        if inside:
            out.append(self.sym[KEEP_END])
        return out

    def encode_source(self, text, tgt_lang, ctx=None, domain=None,
                      protect=True, max_len=None, keep_spans=None):
        """Kaynak dizisi: <2xx> [ctx… <ctx>] [<domain>] metin </s>

        keep_spans: sözlük enjeksiyonundan gelen korunacak karakter aralıkları
        (bkz. glossary.Glossary.rewrite).
        """
        ids = [self.lang_id(tgt_lang)]
        if ctx:
            for c in (ctx if isinstance(ctx, (list, tuple)) else [ctx]):
                ids += self.sp.encode(c, out_type=int)
            ids.append(self.sym[CTX_TAG])
        if domain:
            tag = f"<{domain}>"
            if tag not in self.sym:
                raise KeyError(f"bilinmeyen domain: {domain} ({DOMAIN_TAGS})")
            ids.append(self.sym[tag])
        if protect or keep_spans:
            ids += self._encode_protected(text, extra_spans=keep_spans)
        else:
            ids += self.sp.encode(text, out_type=int)
        ids.append(self.eos_id)
        if max_len and len(ids) > max_len:
            # Baştaki etiketleri KORU, bağlamı/metni sondan kırp
            ids = ids[:max_len - 1] + [self.eos_id]
        return ids

    def encode_target(self, text, protect=True, max_len=None):
        """Hedef dizisi: metin </s>  (decoder girdisi dataset'te <s> ile kaydırılır)"""
        ids = self._encode_protected(text) if protect else self.sp.encode(text, out_type=int)
        ids.append(self.eos_id)
        if max_len and len(ids) > max_len:
            ids = ids[:max_len - 1] + [self.eos_id]
        return ids

    # ----------------------------- DECODE -----------------------------
    def decode(self, ids, strip_special=True):
        """id listesi -> metin. Özel tokenlar temizlenir, <nl> satır sonuna döner."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out, nl = [], self.sym[NEWLINE_TAG]
        drop = {self.pad_id, self.bos_id, self.eos_id}
        drop |= {self.sym[s] for s in USER_SYMBOLS if s != NEWLINE_TAG}
        buf = []
        for i in ids:
            if i == self.eos_id:
                break
            if i == nl:
                out.append(self.sp.decode(buf))
                buf = []
                continue
            if strip_special and i in drop:
                continue
            buf.append(i)
        out.append(self.sp.decode(buf))
        return "\n".join(out)

    def fertility(self, lines):
        """token/kelime oranı. TR ~1.6-2.0, EN ~1.2-1.4 sağlıklı sayılır."""
        n_tok = n_word = 0
        for ln in lines:
            n_tok += len(self.sp.encode(ln, out_type=int))
            n_word += max(1, len(ln.split()))
        return n_tok / max(1, n_word)


if __name__ == "__main__":
    import sys
    tok = VexiraTokenizer(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    print(f"vocab={tok.vocab_size}  model={tok.model_path}")
    print("özel token id'leri:")
    for s in USER_SYMBOLS:
        print(f"  {s:16s} -> {tok.sym[s]}")

    samples = [
        ("Hello {player_name}, you have %d items.", "tr"),
        ("Merhaba, bugün hava çok güzel.", "en"),
        ("<color=#ff0000>Warning</color>: press [b]ESC[/b] to quit.", "tr"),
    ]
    for text, lang in samples:
        ids = tok.encode_source(text, lang)
        print(f"\n  {text!r} -> <2{lang}>")
        print(f"    {len(ids)} token: {[tok.sp.id_to_piece(i) for i in ids]}")
        print(f"    round-trip: {tok.decode(ids)!r}")
