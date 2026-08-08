# -*- coding: utf-8 -*-
"""
Vexira sözlüğü (termbase) — çeviri tutarlılığı için.

NEDEN GEREKLİ
    Sinir ağı her cümleyi sıfırdan karar vererek çevirir; aynı terim iki farklı
    cümlede iki farklı karşılık alabilir ("renderer" -> bir yerde "oluşturucu",
    başka yerde "renderer"). Bu kalite değil TUTARLILIK sorunudur ve daha fazla
    eğitim verisiyle çözülmez — Google/DeepL de model tarafında çözmez, sözlük
    (glossary/termbase) katmanı ekler. Vexira'da da öyle.

İKİ MEKANİZMA
    1) TAM EŞLEŞME (exact)  — girdinin TAMAMI sözlükte varsa model hiç
       çağrılmaz, karşılık doğrudan döner. Deterministik, %100 tutarlı.
       Arayüz etiketleri ("Auto Save", "Page {}") tam olarak bu sınıfa girer.
    2) ENJEKSİYON (inject)  — terim uzun bir cümlenin İÇİNDE geçiyorsa kaynak
       metindeki terim, hedef karşılığıyla değiştirilip <keep_start>..<keep_end>
       arasına alınır. Model bu bölgeyi "aynen taşı" olarak öğrenmiştir, dolayısıyla
       karşılığı olduğu gibi çıkarır ve gerekirse Türkçe ekini sonuna ekler.
       (WMT terminology task yaklaşımı; kısıtlı beam search'ten daha az akıcılık
       bozar.)

BEYNE GÖMME
    Sözlük checkpoint'in İÇİNE yazılabilir (`ck["glossary"]`). Böylece modeli
    taşıyan tek dosya kalır: vexira.pt = ağırlık + config + adım + SÖZLÜK.
    Tokenizer .model dosyası ayrı kalmak zorunda (SentencePiece kendi ikili
    formatını istiyor), ama o da checkpoint'e gömülü tutulur ve gerekirse
    `--extract-spm` ile geri yazılır.

KULLANIM
    python glossary.py build   terms.tsv --out models/vexira.pt   # beyne göm
    python glossary.py show    models/vexira.pt
    python glossary.py export  models/vexira.pt --out terms.tsv
    python glossary.py test    models/vexira.pt

TSV BİÇİMİ (sekmeyle ayrılmış — terimlerde "-" ve ":" geçtiği için virgül/
iki nokta AYIRAÇ DEĞİL):
    en<TAB>tr[<TAB>domain][<TAB>inject]
    Auto Save<TAB>Otomatik Kayıt<TAB>ui
    renderer<TAB>oluşturucu<TAB>ui<TAB>1
    play<TAB>oynat<TAB>ui<TAB>0        # 0 = cümle içine enjekte etme (çok anlamlı)
"""

import argparse
import json
import os
import re
import sys

TR_UP = str.maketrans("iıçğöşü", "İIÇĞÖŞÜ")
TR_LO = str.maketrans("IİÇĞÖŞÜ", "ıiçğöşü")


def tr_upper(s):
    return s.translate(TR_UP).upper()


def tr_lower(s):
    return s.translate(TR_LO).lower()


def norm_key(s):
    """Eşleşme anahtarı: Türkçe-duyarlı küçük harf + boşluk sadeleştirme."""
    return re.sub(r"\s+", " ", tr_lower(s)).strip()


def tr_title(s):
    """Her kelimenin ilk harfini Türkçe kurallarıyla büyüt (i -> İ, ı -> I).
    Kelimenin geri kalanına DOKUNMA: sözlükteki yazım zaten doğru
    ("Otomatik Kayıt" -> "Otomatik Kayıt", "USB kablosu" -> "USB Kablosu")."""
    return " ".join((w[:1].translate(TR_UP).upper() + w[1:]) if w else w
                    for w in s.split(" "))


def match_case(src, tgt):
    """Kaynağın büyük/küçük harf desenini hedefe taşı.

    Arayüz metninde bu görünürden önemli: aynı etiket bir yerde "Metin Hızı"
    diğerinde "metin hızı" çıkarsa tutarsızlık gözle görülür.

      "AUTO SAVE" -> "OTOMATİK KAYIT"   (Türkçe i -> İ)
      "Text Speed" -> "Metin Hızı"      (başlık durumu korunur)
      "auto save"  -> "otomatik kayıt"
      "Auto save"  -> hedef aynen (sözlükteki yazım)
    """
    letters = [c for c in src if c.isalpha()]
    if not letters:
        return tgt
    if all(c.isupper() for c in letters) and len(letters) > 1:
        return tr_upper(tgt)
    if all(c.islower() for c in letters):
        return tr_lower(tgt)
    words = [w for w in src.split() if w and w[0].isalpha()]
    if words and all(w[0].isupper() for w in words):
        # Tek kelime de dahil. Önce yalnız çok kelimeliye bakıyordu; "Save"
        # girdisi sözlükteki "KAYDET" yazımını olduğu gibi döndürüyordu.
        return tr_title(tgt)
    return tgt


class Glossary:
    """Çift yönlü terim sözlüğü. Yön anahtarı hedef dil: "tr" (en->tr) / "en" (tr->en)."""

    def __init__(self, entries=None, allow_inject=False):
        """allow_inject: cümle içi enjeksiyonu aç.

        VARSAYILAN KAPALI — ölçüldü. Enjeksiyon, modelin <keep_start>..<keep_end>
        bölgesini aynen taşıyacağına güvenir. Ön-eğitimde bu davranış zayıf
        öğrenilmiş (arayüz verisi toplamın %0.18'i) ve 559 satırlık gerçek testte
        model koruma bölgesini bozdu:
            "Clipboard voicing enabled."
              enjekte edilen kaynak: "Pano seslendirme enabled."
              model çıktısı        : "[Bölge seslendirme] etkin."   <- uymadı
        Tam eşleşme ise modele hiç sormaz, bozulamaz. İnce ayar (SFT) enjeksiyon
        biçimini öğrettikten SONRA allow_inject=True yapılacak.
        """
        self.entries = list(entries or [])
        self.allow_inject = allow_inject
        self._build()

    # ------------------------------------------------------------------ kurulum
    def _build(self):
        self.exact = {"tr": {}, "en": {}}
        self._inject = {"tr": [], "en": []}
        for e in self.entries:
            en, tr = e.get("en", "").strip(), e.get("tr", "").strip()
            if not en or not tr:
                continue
            dom = e.get("domain") or None
            inj = bool(e.get("inject", True))
            for tgt_lang, s, t in (("tr", en, tr), ("en", tr, en)):
                key = norm_key(s)
                if not key:
                    continue
                self.exact[tgt_lang].setdefault(key, (t, dom))
                if inj and len(key) >= 4:      # 3 harften kısa terim cümle içinde çok riskli
                    self._inject[tgt_lang].append((s, t, dom))
        # uzun terim önce eşleşmeli ("save slot" , "save"den önce)
        for k in self._inject:
            self._inject[k].sort(key=lambda x: -len(x[0]))
            self._inject[k] = [(self._compile(s), s, t, d)
                               for s, t, d in self._inject[k]]

    @staticmethod
    def _compile(term):
        """Kelime sınırlı, harf-durumu duyarsız desen. Terim içi boşluk esnek."""
        parts = [re.escape(p) for p in term.split()]
        body = r"\s+".join(parts)
        # \b latin harfleriyle çalışır; Türkçe harfler \w içinde (re.UNICODE varsayılan)
        return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)

    def __len__(self):
        return len(self.entries)

    def counts(self):
        return {"terim": len(self.entries),
                "en->tr tam": len(self.exact["tr"]),
                "tr->en tam": len(self.exact["en"]),
                "en->tr enjekte": len(self._inject["tr"]),
                "tr->en enjekte": len(self._inject["en"])}

    # ------------------------------------------------------------------ arama
    def lookup(self, text, tgt_lang, domain=None):
        """Girdinin TAMAMI sözlükteyse karşılığını döndür, yoksa None.

        domain verilmişse: sözlükte domain'i olan kayıt yalnız o domain'de geçerli.
        Domain'siz kayıt her yerde geçerli.
        """
        hit = self.exact.get(tgt_lang, {}).get(norm_key(text))
        if hit is None:
            return None
        tgt, dom = hit
        if dom and domain and dom != domain:
            return None
        return match_case(text.strip(), tgt)

    def rewrite(self, text, tgt_lang, domain=None, max_terms=8):
        """Cümle içi terimleri hedef karşılığıyla değiştir.

        Döner: (yeni_metin, korunacak_karakter_aralıkları, kaç_terim).
        Aralıklar tokenizer'a verilir; oradaki tokenlar <keep_start>..<keep_end>
        arasına alınır ve model o bölgeyi aynen taşır.
        """
        if not self.allow_inject:
            return text, [], 0
        spans_src = []          # (başlangıç, bitiş, karşılık) — orijinal metinde
        taken = []
        for rx, _s, t, dom in self._inject.get(tgt_lang, []):
            if len(spans_src) >= max_terms:
                break
            if dom and domain and dom != domain:
                continue
            for m in rx.finditer(text):
                if any(m.start() < e and m.end() > b for b, e in taken):
                    continue        # daha uzun bir terim burayı zaten aldı
                taken.append((m.start(), m.end()))
                spans_src.append((m.start(), m.end(), match_case(m.group(0), t)))
                if len(spans_src) >= max_terms:
                    break
        if not spans_src:
            return text, [], 0

        spans_src.sort()
        out, keep, pos = [], [], 0
        for b, e, rep in spans_src:
            out.append(text[pos:b])
            start = sum(len(x) for x in out)
            out.append(rep)
            keep.append((start, start + len(rep)))
            pos = e
        out.append(text[pos:])
        return "".join(out), keep, len(spans_src)

    # ------------------------------------------------------------------ G/Ç
    @classmethod
    def from_tsv(cls, path):
        entries = []
        with open(path, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                p = line.split("\t")
                if len(p) < 2 or not p[0].strip() or not p[1].strip():
                    print(f"  ! {path}:{ln} atlandı (2 sütun gerekli): {line[:60]!r}")
                    continue
                e = {"en": p[0].strip(), "tr": p[1].strip()}
                if len(p) > 2 and p[2].strip():
                    e["domain"] = p[2].strip()
                if len(p) > 3 and p[3].strip():
                    e["inject"] = p[3].strip() not in ("0", "no", "hayır")
                entries.append(e)
        return cls(entries)

    def to_tsv(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# en\ttr\tdomain\tinject\n")
            for e in self.entries:
                f.write(f"{e['en']}\t{e['tr']}\t{e.get('domain','')}\t"
                        f"{1 if e.get('inject', True) else 0}\n")
        return len(self.entries)

    def to_dict(self):
        return {"version": 1, "entries": self.entries}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls([])
        return cls(d.get("entries", []))

    # -------------------------------------------------- checkpoint'e göm / oku
    @classmethod
    def from_ckpt(cls, path):
        """Checkpoint içindeki sözlüğü oku. Yoksa boş sözlük döner."""
        import torch
        if not os.path.exists(path):
            return cls([])
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False,
                            mmap=True)
        except Exception:                                        # noqa: BLE001
            ck = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_dict(ck.get("glossary"))

    def embed(self, ckpt_path, spm_path=None):
        """Sözlüğü (ve istenirse tokenizer'ı) checkpoint'in İÇİNE yaz.

        Ağırlıklara dokunmaz; sadece üst seviye anahtar ekler. Atomik yazım.
        """
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ck["glossary"] = self.to_dict()
        if spm_path:
            with open(spm_path, "rb") as f:
                ck["spm_model"] = f.read()
            ck["spm_sha"] = _sha(spm_path)
        tmp = ckpt_path + ".tmp"
        torch.save(ck, tmp)
        os.replace(tmp, ckpt_path)
        return os.path.getsize(ckpt_path)


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def extract_spm(ckpt_path, out_path):
    """Checkpoint'e gömülü tokenizer'ı diske yaz (model tek dosya dağıtılırsa)."""
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    blob = ck.get("spm_model")
    if not blob:
        return None
    with open(out_path, "wb") as f:
        f.write(blob)
    return out_path


# ------------------------------------------------------------------ CLI

def _cmd_build(a):
    g = Glossary.from_tsv(a.tsv)
    print(f"{a.tsv}: {len(g)} terim okundu")
    for k, v in g.counts().items():
        print(f"  {k:16s} {v}")
    if a.out:
        before = os.path.getsize(a.out)
        size = g.embed(a.out, spm_path=a.spm)
        print(f"-> {a.out} gömüldü ({before/1e6:.1f} MB -> {size/1e6:.1f} MB)")
    return 0


def _cmd_show(a):
    g = Glossary.from_ckpt(a.ckpt)
    if not len(g):
        print(f"{a.ckpt}: gömülü sözlük YOK")
        return 1
    for k, v in g.counts().items():
        print(f"  {k:16s} {v}")
    print("\nilk 20:")
    for e in g.entries[:20]:
        print(f"  {e['en']:34s} -> {e['tr']:34s} {e.get('domain','')}")
    return 0


def _cmd_export(a):
    g = Glossary.from_ckpt(a.ckpt)
    n = g.to_tsv(a.out)
    print(f"{n} terim -> {a.out}")
    return 0


def _words(s):
    return [w for w in re.sub(r"[^\w\s]", " ", tr_lower(s)).split() if w]


def truncated_vs(gloss_tr, model_tr):
    """Sözlük karşılığı, modelin çevirisinin KIRPILMIŞ hâli mi?

    Gerçek vaka: "deactivate" için sözlük "Devre", modelin kendi çevirisi
    "devre dışı bırak". Sözlük burada modelden KÖTÜ — böyle bir giriş
    tutarlılık kazandırmaz, doğrudan hata üretir. Tek oyla gelen girişlerin
    tipik kusuru bu.

    Ölçüt üç koşulun HEPSİ: sözlük TEK kelime · model çıktısı 3+ kelime ·
    model çıktısı o kelimeyle başlıyor.

    İlk hâli yalnız "ön ek + model daha uzun" idi; "Cancel" için sözlük 'İptal',
    model 'İptal Et' çıkınca doğru girişi de atıyordu. Tek fazla kelime bir
    üslup farkıdır; İKİ fazla kelime "devre" -> "devre dışı bırak" gibi anlamı
    tamamen değiştiren sabit bir öbeğe işaret ediyor.

    "görünüm alanı" vs "görünüm portu" zaten ön ek değil (ikinci kelime farklı) —
    orada sözlük bilinçli bir tercih, korunur.
    """
    g, m = _words(gloss_tr), _words(model_tr)
    return len(g) == 1 and len(m) >= 3 and m[0] == g[0]


def _cmd_validate(a):
    """Sözlüğü modelin kendi çevirisiyle karşılaştır, kırpık girişleri at."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from translate import Translator                               # noqa: E402
    g = Glossary.from_tsv(a.tsv) if a.tsv else Glossary.from_ckpt(a.ckpt)
    if not len(g):
        print("sözlük boş")
        return 1
    tr = Translator(a.ckpt, a.spm, a.device, use_glossary=False)
    srcs = [e["en"] for e in g.entries]
    outs = tr.translate(srcs, "tr", domain=a.domain, beam=a.beam)

    keep, drop = [], []
    for e, mo in zip(g.entries, outs):
        (drop if truncated_vs(e["tr"], mo) else keep).append((e, mo))
    print(f"{len(g)} terim · {len(keep)} tutuldu · {len(drop)} KIRPIK atıldı")
    for e, mo in drop:
        print(f"    {e['en']!r}: sözlük {e['tr']!r} <- model {mo!r} (model daha tam)")
    if a.out:
        Glossary([e for e, _ in keep]).to_tsv(a.out)
        print(f"-> {a.out}")
    return 0


def _cmd_test(a):
    """Kendi kendini doğrulayan birim test — sözlük mantığı bozulmasın."""
    g = Glossary(allow_inject=True, entries=[
        {"en": "Auto Save", "tr": "Otomatik Kayıt", "domain": "ui"},
        {"en": "renderer", "tr": "oluşturucu", "domain": "ui"},
        {"en": "text speed", "tr": "metin hızı", "domain": "ui"},
        {"en": "save slot", "tr": "kayıt yeri", "domain": "ui"},
        {"en": "save", "tr": "kaydet", "domain": "ui"},
        {"en": "play", "tr": "oynat", "domain": "ui", "inject": False},
    ])
    ok = fail = 0

    def chk(name, got, want):
        nonlocal ok, fail
        good = got == want
        ok, fail = ok + good, fail + (not good)
        print(f"  {'✓' if good else '✗'} {name}")
        if not good:
            print(f"      beklenen {want!r}\n      gelen    {got!r}")

    chk("tam eşleşme", g.lookup("Auto Save", "tr"), "Otomatik Kayıt")
    chk("tam eşleşme boşluk/harf", g.lookup("  auto   save ", "tr"), "otomatik kayıt")
    chk("tam eşleşme BÜYÜK", g.lookup("AUTO SAVE", "tr"), "OTOMATİK KAYIT")
    chk("başlık durumu", g.lookup("Text Speed", "tr"), "Metin Hızı")
    chk("karışık durum -> sözlük yazımı", g.lookup("Auto save", "tr"), "Otomatik Kayıt")
    chk("tek kelime Başlık -> Başlık", g.lookup("Renderer", "tr"), "Oluşturucu")
    chk("tek kelime küçük -> küçük", g.lookup("renderer", "tr"), "oluşturucu")
    chk("ters yön", g.lookup("Otomatik Kayıt", "en"), "Auto Save")
    chk("sözlükte yok", g.lookup("Quit game", "tr"), None)
    chk("domain uyuşmazlığı", g.lookup("Auto Save", "tr", domain="sub"), None)
    chk("domain uyuşması", g.lookup("Auto Save", "tr", domain="ui"), "Otomatik Kayıt")

    t, keep, n = g.rewrite("Change the text speed in settings.", "tr", domain="ui")
    chk("cümle içi enjeksiyon", (t, n), ("Change the metin hızı in settings.", 1))
    chk("korunan aralık", t[keep[0][0]:keep[0][1]], "metin hızı")

    t2, k2, n2 = g.rewrite("Use save slot 3 to save.", "tr", domain="ui")
    chk("uzun terim önce", t2, "Use kayıt yeri 3 to kaydet.")
    chk("iki terim korunur", len(k2), 2)
    chk("aralıklar doğru", [t2[b:e] for b, e in k2], ["kayıt yeri", "kaydet"])

    t3, _, n3 = g.rewrite("They play the piano.", "tr", domain="ui")
    chk("inject=0 uygulanmadı", n3, 0)

    t4, _, n4 = g.rewrite("The renderers were slow.", "tr", domain="ui")
    chk("kelime sınırı (renderers != renderer)", n4, 0)

    off = Glossary(g.entries)                     # allow_inject varsayılan False
    chk("enjeksiyon varsayılan KAPALI",
        off.rewrite("Change the text speed now.", "tr", domain="ui")[2], 0)
    chk("kapalıyken tam eşleşme ÇALIŞIR",
        off.lookup("Auto Save", "tr"), "Otomatik Kayıt")

    print(f"\n{ok} geçti, {fail} kaldı")
    return 1 if fail else 0


def main():
    ap = argparse.ArgumentParser(description="Vexira sözlük aracı")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="TSV -> checkpoint'e göm")
    b.add_argument("tsv")
    b.add_argument("--out", default="models/vexira.pt")
    b.add_argument("--spm", default=None, help="tokenizer'ı da göm (tek dosya dağıtım)")
    b.set_defaults(fn=_cmd_build)

    s = sub.add_parser("show", help="gömülü sözlüğü göster")
    s.add_argument("ckpt", nargs="?", default="models/vexira.pt")
    s.set_defaults(fn=_cmd_show)

    e = sub.add_parser("export", help="gömülü sözlük -> TSV")
    e.add_argument("ckpt", nargs="?", default="models/vexira.pt")
    e.add_argument("--out", default="terms.tsv")
    e.set_defaults(fn=_cmd_export)

    x = sub.add_parser("extract-spm", help="gömülü tokenizer'ı diske yaz")
    x.add_argument("ckpt", nargs="?", default="models/vexira.pt")
    x.add_argument("--out", default="models/vexira_spm.model")
    x.set_defaults(fn=lambda a: (print(extract_spm(a.ckpt, a.out) or "gömülü spm yok"), 0)[1])

    v = sub.add_parser("validate",
                       help="sözlüğü modele karşı doğrula, kırpık girişleri at")
    v.add_argument("--ckpt", default="models/vexira.pt")
    v.add_argument("--tsv", default=None, help="yoksa ckpt'ye gömülü sözlük")
    v.add_argument("--spm", default=None)
    v.add_argument("--device", default="cpu")
    v.add_argument("--domain", default="ui")
    v.add_argument("--beam", type=int, default=4)
    v.add_argument("--out", default=None, help="temizlenmiş TSV yolu")
    v.set_defaults(fn=_cmd_validate)

    t = sub.add_parser("test", help="birim test")
    t.set_defaults(fn=_cmd_test)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
