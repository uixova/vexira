# Vexira

**80.8M parametreli TR↔EN çeviri modeli.** Tamamen yerel çalışır — internet,
API anahtarı, hesap gerekmez. Altyazı, oyun metni, arayüz dizeleri ve dosya
içeriği için tasarlandı.

[English](README_EN.md) · [Hugging Face](https://huggingface.co/uixova/vexira) · [GitHub](https://github.com/uixova/vexira)

| | |
|---|---|
| Parametre | 80.8M (fp16, ~490 MB) |
| Diller | Türkçe ↔ İngilizce, çift yönlü, tek model |
| Hız | **78 ms/satır** (12.9 satır/sn) — 4 CPU çekirdeği, GPU gerekmez |
| FLORES-200 | en→tr **BLEU 29.1 / chrF++ 57.7** · tr→en **BLEU 35.2 / chrF++ 60.1** |
| Eğitim | 7.56B token (4.67× Chinchilla) + arayüz odaklı ince ayar |
| Lisans | Apache-2.0 |

---

## Hızlı başlangıç

```bash
git clone https://github.com/uixova/vexira && cd vexira
pip install torch sentencepiece

# Ağırlıkları indir (~490 MB)
huggingface-cli download uixova/vexira vexira_sft.pt vexira_spm.model --local-dir models/
```

**Terminal bilmiyorsan** — çift tıkla, menü açılır:

```
./vexira.sh      Linux / macOS   — çift tık, kendi terminalini açar
vexira.bat       Windows         — çift tık
vexira.ps1       Windows         — sağ tık > "Run with PowerShell"
```

> `.ps1` çift tıklanınca Not Defteri'nde açılır (Windows'un varsayılanı).
> Çift tıklama istiyorsan `vexira.bat` kullan.

```
 1  Metin çevir   EN → TR       4  Dosya çevir   TR → EN
 2  Metin çevir   TR → EN       5  Tarayıcı arayüzü
 3  Dosya çevir   EN → TR       q  çıkış
```

Çıktı hem ekrana basılır hem diske yazılır (`ceviriler/` altına, dosya
çevirisinde kaynağın yanına `dosya.tr.txt` olarak). Model bir kez yüklenir,
menü boyunca bellekte kalır.

Komut satırını tercih edersen tek satır:

```bash
# Metin
python translate.py --text "Hello world" --to tr
# -> Merhaba dünya

# Dosya
python translate.py --file samples/sample_en.txt --to tr --domain doc

# Altyazı (.srt zaman kodlarını korur)
python translate.py --srt film.srt --to tr --out film.tr.srt

# Kalıcı servis — OCR/TTS/STT katmanları için, model bir kez yüklenir
python translate.py --server
```

**Tarayıcı arayüzü** — tek komut, yeni bağımlılık yok (stdlib `http.server`):

```bash
python webui.py            # http://127.0.0.1:8770 kendiliğinden açılır
```

Sol tarafa yaz, sağdan kopyala. Yön/domain/beam seçici, `Ctrl+Enter` kısayolu,
canlı istatistik. HTML tek dosyada gömülü — internetsiz makinede de açılır,
yalnız `127.0.0.1`e bağlanır.

Python'dan:

```python
from translate import Translator

tr = Translator()                      # models/vexira_sft.pt otomatik bulunur
print(tr.translate(["Save", "Are you sure you want to quit?"],
                   tgt_lang="tr", domain="ui"))
# ['Kaydet', 'Çıkmak istediğinizden emin misiniz?']
```

### "Bu kadar kod indirmek zorunda mıyım?" — hayır

`translate.py` 650+ satır çünkü tam bir **araç**: sözlük, `.srt`, sunucu modu,
uzun girdi bölme, çıktı onarımı. Modeli **çalıştırmak** için o kadarı gerekmez.
[`examples/minimal.py`](examples/minimal.py) çekirdeğin tamamı — 40 satır:

```python
ck = torch.load("models/vexira_sft.pt", map_location="cpu", weights_only=False)
torch.set_num_threads(ck["runtime"]["threads"])        # ayar modelin İÇİNDE
cfg = VexiraConfig.from_dict(ck["config"]); cfg.dropout = 0.0
model = Vexira(cfg); model.load_state_dict(ck["model"]); model.eval()

src    = torch.tensor([tok.encode_source(text, "tr", max_len=128)])
mem    = model.encode(src)                             # encoder bir kez
caches = model.init_cache(mem, max_len=cfg.max_pos)    # KV cache
# ... greedy döngü ...
```

```bash
python examples/minimal.py "Hello world, this is a small translation model."
# -> Merhaba dünya, bu küçük bir çeviri modeli.
```

Farkı görmek için aynı cümle iki yoldan:

```
minimal.py    "The renderer failed to start." -> "Tezgah başlatılamadı."      ✗
translate.py  (sözlük + --domain ui)          -> "Oluşturucu başlatılamadı."  ✓
```

Yani minimal sürüm "modeli nasıl çağırırım"ın cevabı; kaliteyi sözlük katmanı
veriyor.

### Kendi projene bağlama

Vexira'yı tek başına kullanmak yerine kendi boru hattına (başka bir LLM, OCR,
altyazı aracı) takacaksan `translate.py`'nin 650 satırını okuman gerekmez.
Ondan aldığın tek şey `Translator`:

```python
from translate import Translator

tr = Translator()                                    # modeli BİR KEZ yükle
out = tr.translate(lines, tgt_lang="tr", domain="ui")
```

Bu üç satır her şeyi getirir: sözlük, yer tutucu onarımı, tekrar sadeleştirme,
128 token üstü bölme, toplu iş, doğru thread sayısı. **`Translator` entegrasyon
arayüzüdür**; `examples/minimal.py` ise "içeride ne oluyor"u gösteren öğretici
sürüm — üretimde onu kullanma, onarımlar orada yok.

Çalışan tam örnek: [`examples/integrate.py`](examples/integrate.py) — toplu iş,
domain seçimi, istatistik, kendi terim sözlüğün, kural okuma, Python dışı
kullanım.

**Onarım kuralları da modelin içinde.** Kusurlar bu modelin eğitim verisinden
geliyor, o yüzden kurallar da modelle taşınıyor:

```python
tr.pp                 # checkpoint'ten gelen kurallar
ck["postprocess"]     # ham hâli — JSON, Rust/JS/C++ tarafından da okunabilir
```

Böylece modeli başka bir dilde çalıştıran kişi aynı kusurları yeniden
keşfetmek zorunda kalmaz.

**Python dışından** — `--server` alt süreç olarak çağrılır, model bir kez
yüklenir:

```bash
python translate.py --server
# {"lines":["Hello"],"to":"tr","domain":"ui"}
# -> {"ok":true,"out":["Merhaba"],"ms":23.1}
```

**Hız için tek kural:** satırları döngüde tek tek değil, **tek çağrıda** ver.
139 ms/satır yerine 62 ms/satır.

### Çalışma ayarları modelin içinde

Doğru thread/beam değerleri modele özgü ve ölçümle bulundu — kullanıcının bunu
bilmesi beklenemez, o yüzden **checkpoint'in içine gömülü**:

```python
ck["runtime"]   # {'threads': 4, 'beam': 2, 'batch_size': 32, 'domain': 'sub'}
```

Model nereye giderse ayarı da yanında gider. Geçersiz kılma sırası:

```
CLI bayrağı  >  ortam değişkeni  >  modele gömülü  >  kod varsayılanı
--threads 8     VEXIRA_THREADS=8   ck["runtime"]      RUNTIME_DEFAULTS
```

```bash
VEXIRA_THREADS=2 VEXIRA_BEAM=4 python translate.py --text "..." --to tr
```

### Kendi verinle dene

`samples/` altında iki dosya var — kolaydan zora sıralı, modelin nerede
zorlandığını görürsün:

```bash
python translate.py --file samples/sample_en.txt --to tr --domain doc   # EN -> TR
python translate.py --file samples/sample_tr.txt --to en --domain doc   # TR -> EN
```

İçerik: kısa arayüz etiketleri, yer tutuculu metin, altyazı satırları,
çok anlamlı kelimeler, düzyazı, teknik dil, 128 token üstü uzun paragraf,
devrik ve deyimli yapılar.

### Önemli parametreler

| bayrak | varsayılan | not |
|---|---|---|
| `--to` | `tr` | hedef dil: `tr` / `en` |
| `--domain` | `sub` | `sub` altyazı · `ui` arayüz · `doc` düzyazı · `ocr` ekran metni |
| `--beam` | 2 | 4 yaklaşık +0.5 BLEU ama ~1.7× yavaş |
| `--threads` | oto (4) | **artırma** — bu boyutta fazla thread yavaşlatır, ölçüm aşağıda |
| `--no-glossary` | kapalı | sözlüğü devre dışı bırak (ham model çıktısı) |

`--domain` gerçekten fark eder: `ui` verildiğinde sözlük devreye girer ve
`Save` → `Kaydet` deterministik olur; `doc` verildiğinde model kendi kararını
kullanır.

---

## Ölçümler

### FLORES-200 devtest (1012 cümle, beam 4, sözlük kapalı)

| model | en→tr BLEU | chrF++ | tr→en BLEU | chrF++ |
|---|---|---|---|---|
| `vexira.pt` (ön-eğitim) | **29.53** | 57.84 | 35.03 | 59.98 |
| **`vexira_sft.pt`** (ana) | 29.13 | 57.68 | **35.16** | **60.11** |

Karşılaştırma için aynı test setinde bilinen değerler: NLLB-200 distilled
(600M) en→tr ~26-28, opus-mt-tc-big-en-tr (230M) ~30. Vexira 80.8M ile bu
bandın içinde.

> tr→en'in daha yüksek görünmesi model farkı değil metrik etkisi: BLEU kelime
> eşleşmesine bakar, Türkçe'nin çekim ekleri yüzünden `evden`/`eve` tam yanlış
> sayılır. chrF++ farkı (60.1 vs 57.7) gerçek farkı daha iyi gösterir.

### Terim tutarlılığı — 559 satırlık gerçek Ren'Py dosyası

BLEU'nun göremediği kusur: aynı terimin farklı satırlarda farklı çevrilmesi.

| model | tutarsız terim | sözlükle |
|---|---|---|
| `vexira.pt` | 33 | 29 |
| **`vexira_sft.pt`** | **19** | 21 |

İnce ayarın asıl kazancı burada: **%42 azalma**. Ayrıca sözlük 559 dizenin
**%42'sini** tam eşleşmeyle çeviriyor — o satırlarda model hiç çağrılmıyor,
tutarsızlık matematiksel olarak imkânsız.

### Hız (12 çekirdekli dizüstü, CPU, GPU yok)

| thread | beam 1 | beam 2 | beam 4 |
|---|---|---|---|
| 2 | 76.8 ms | 105.8 ms | 162.3 ms |
| **4** | 82.5 ms | **77.6 ms** | 135.5 ms |
| 8 | 86.7 ms | 133.5 ms | 181.6 ms |
| 12 | 338.9 ms | 466.0 ms | 367.7 ms |

**Thread sayısını artırmak yavaşlatıyor.** Bu boyutta katman başına iş küçük,
senkronizasyon maliyeti hesabı geçiyor. `translate.py` bu yüzden varsayılanı
torch'a bırakmıyor, 4 ile sınırlıyor — kutudan çıktığı hâliyle 6× fark ediyor.

---

## Nasıl çalışıyor

### Mimari

| | değer |
|---|---|
| tip | encoder-decoder (decoder-only değil — çeviride parametre verimliliği kat kat yüksek) |
| encoder / decoder | 12 / 6 |
| d_model / d_ff / head | 512 / 1408 / 8 |
| vocab | 32000 ortak TR+EN SentencePiece **unigram** |
| **etkin bağlam** | **128 token** (~80-100 kelime) — [ayrıntı](#etkin-bağlam-128-token) |
| pozisyon | öğrenilmiş mutlak (RoPE değil) |
| norm / ffn | RMSNorm (pre-LN) / SwiGLU |
| embedding | tied (encoder + decoder + lm_head tek tablo) |
| toplam | **80.76M** — embed 16.4M · enc 38.5M · dec 25.6M |

**Neden encoder-decoder.** Çeviri, kaynağı tam görüp hedefi üretmek demek.
Decoder-only mimaride aynı kaliteyi yakalamak kat kat daha fazla parametre ister.

**Neden 12+6.** Uçtan uca gecikmede encoder baskın: 12 katman × 48 kaynak token'ı
tek seferde işliyor, decoder 6 katman × ~20 adım. Decoder'ı 4→6 çıkarmak toplam
maliyeti %6 artırıyor; encoder'dan 2 katman alıp decoder'a vermek modeli **%5
hızlandırıyor** ve Türkçe çekim eki üretimini sağlamlaştırıyor.

**Anlam nerede.** Embedding tablosu (16.4M) bağlamdan bağımsız statik lookup —
anlam orada değil. `bank` tabloda tek satır; 12 encoder katmanından sonra
"river bank" ve "bank account" farklı vektörler:

```
He works at the bank.               -> Bankada çalışıyor.
I went to the river bank to think.  -> Düşünmek için nehir kıyısına gittim.
She plays the piano every evening.  -> Her akşam piyano çalıyor.
The children play in the garden.    -> Çocuklar bahçede oynuyor.
```

Aynı sınıftaki klasik NMT modelleri (~77M) vocab'ı 58-65k olduğu için bütçenin
%43'ünü embedding'e yakıyor; Vexira'da bu oran %21 — ~17M parametre daha fazlası
gerçek katmanlarda.

**Sıkıştırma yok.** int8/ternary bilinçli olarak yok: bu ölçekte kalite kaybı
maliyetine değmiyor. Hız KV cache, tek-sefer encoder K/V ve thread sınırından.

### Etkin bağlam: 128 token

`config.max_pos = 512` ama **kullanılabilir bağlam 128 token**. Ön-eğitim
`max_len=128` ile yapıldı; pozisyon gömmelerinin 128-511 arası hiç güncellenmedi:

```
pos   0-127 : norm 9.5 – 10.4     eğitilmiş
pos 128-511 : norm 0.34           ilk değerinde
```

`Translator` bu tavanı **ölçerek** buluyor, config'e güvenmiyor. Aşan girdi
cümle sınırından bölünüyor, parçalar `<ctx>` ile zincirleniyor, çıktı
birleştiriliyor — Google'ın yaptığının aynısı. Ölçüm: 144 tokenlik 16 cümlede
**16/16 kavram korundu**.

512'ye çıkarmak anlamlı değil: temiz korpusun yalnız **%0.31'i** 128 token
üstünde. Paralel korpuslar tanımı gereği cümle hizalı; Marian/OPUS-MT de cümle
seviyesi çalışır. Yarı eğitilmiş pozisyon, net bir sınırdan daha kötüdür.

### Terim tutarlılığı — sözlük katmanı

Sinir ağı her cümleyi sıfırdan çevirir, terim hafızası yoktur. Bu kalite değil
**tutarlılık** sorunu ve daha fazla eğitim verisiyle çözülmez — Google ve DeepL
de model tarafında çözmez, sözlük (glossary/termbase) katmanı ekler.

**Sözlük beynin İÇİNDE.** `vexira_sft.pt` = ağırlık + config + adım + **1.234
terim** + tokenizer. Yanına dosya taşımak gerekmez:

```bash
python glossary.py show models/vexira_sft.pt      # içindekini gör
python glossary.py export models/vexira_sft.pt --out terms.tsv
python glossary.py build terms.tsv --out models/vexira_sft.pt --spm models/vexira_spm.model
```

Terimler iki bağımsız kaynaktan geliyor: sistem dil dosyaları
(`/usr/share/locale`, **insan çevirisi**) ve alan terimleri. Bir terim sözlüğe
ancak **≥3 ayrı pakette** geçerse ve o paketlerin **≥%70'i aynı karşılığı**
verirse giriyor — karar bir kişinin görüşü değil, bağımsız çevirmenlerin fiilî
uzlaşması. Uzlaşma yoksa terim alınmıyor, bağlamı encoder çözüyor.

Marka adları asla çevrilmiyor:

```
Windows -> Windows (çevrilmez)     window -> pencere
```

Harf durumu girdiden taşınır: `SAVE`→`KAYDET`, `Save`→`Kaydet`, `save`→`kaydet`.

**Sohbet kısaltmaları.** Eğitim korpusu (altyazı + web) düzgün yazılmış metin;
SMS kısaltması neredeyse hiç geçmiyor. Model tam yazımı biliyor ama kısaltmayı
tanımıyordu:

```
selam knk naber       -> hello knk naber          ✗
selam kanka ne haber  -> Hi, dude, what's up?     ✓
```

Çözüm sözlük değil — sözlük tam eşleşmeyle çalışır, cümle içindeki `knk`'yı
yakalayamaz. Kaynak tarafında **açılıyor**: **64 giriş**, modele gömülü
(`ck["preprocess"]`), yalnız TR→EN yönünde, `--no-expand` ile kapanır.

Tabloda iki kategori var:

**A. Yazım açılımı** — model tam yazımı biliyor, kısaltmayı bilmiyor:

```
selam knk naber  ->  Hi, dude, what's up?      slm nslsn      ->  Hi, how are you?
hersey yolunda   ->  everything is fine        noluyo burda   ->  What's happening here
```

**B. Anlam eşlemesi** — kelimenin *kendisi* modelde yok. Ölçüldü:

```
eyvallah -> "12. 12. 2017"     inşallah -> "Imprint"
maşallah -> "Misha"            yha      -> "xhamster.com"
```

Son satır web korpusundan bulaşma. Bunlar kısaltma değil, kelime dağarcığı
boşluğu; modelin bildiği anlamdaşına eşleniyor:

```
eyvallah -> sağ ol   =>  "Thanks"        inşallah -> umarım  =>  "I hope"
```

Anlam birebir değil ama "12. 12. 2017"ten iyi. İleride veri eklenirse bu
satırlar kaldırılmalı — kodda öyle işaretli.

Her giriş **ölçülerek** seçildi: ham çıktı ile açılmış çıktı karşılaştırıldı,
yalnız iyileştirenler kaldı. `vb -> etc.` zaten doğru olduğu için listeye
alınmadı.

### Çıktı onarımı

Gerçek Ren'Py testinde ölçülen iki kusur, çıkarımda deterministik onarılıyor:

| kusur | örnek | önce | sonra |
|---|---|---|---|
| kaçış dizisi bölünmesi | `\n` → `\ n` | 14 | **0** |
| yer tutucu çevirisi | `[text]` → `[metin]` | 58 | **10** |

`--no-repair` ile kapatılabilir.

---

## Bilinen sınırlar

- **128 token** üstü girdi bölünüp birleştirilir; tek seferde işlenmez.
- **Yalnız TR↔EN.** Başka dil yok.
- **Bilgi modeli değil.** Soru cevaplamaz, sohbet etmez, akıl yürütmez.
- Şiir, kelime oyunu, yoğun deyim ve argo zayıf.
- `<ocr>`, `<stt>`, `<tts>` domain token'ları rezerve ama **eğitilmedi** —
  ileride ince ayarla beslenecek.
- Sözlük arayüz terimlerine odaklı; düzyazıda `--domain doc` ile model kendi
  kararını kullanır.

---

## Eğitim

| aşama | veri | sonuç |
|---|---|---|
| Ön-eğitim | 60.9M çift → 121.7M örnek (iki yön), 5.46B token/epoch | 4 oturum, 7.56B token (4.67× Chinchilla), val 2.6444 |
| İnce ayar | 133.753 örnek, 3 epoch, LR 5e-5 | val 2.4147, 12 dakika |

**Veri kaynakları:** 23 OPUS korpusu (OpenSubtitles, CCMatrix, HPLT, CCAligned,
WikiMatrix, TED, Tatoeba, KDE4, SETIMES …) ve sistem dil dosyaları. FLORES-200
devtest eğitime **hiç girmedi**.

**Epoch politikası:** en fazla 2 epoch. Hacim tekrardan değil taze kaynaktan
geldi — aynı veriyi tekrar göstermek küçük modelde doğrudan ezber.

İnce ayar seti bilinçli dengeli: arayüz %51 / genel %49, uzunluk dağılımı
ön-eğitimle birebir (ortalama 19.4 vs 19.4 token, ≥30 token %17.5 vs %18.1).
Amaç arayüz terimlerini öğretirken genel çeviriyi kaybetmemekti; ölçümler bunun
tuttuğunu gösteriyor.

Eğitim tarafı `training/` altında:

```bash
python training/train_spm.py                    # tokenizer
python training/build_bin.py                    # tokenize -> .bin
python training/train.py --data ... --preset main
python training/train.py --data ... --finetune --init-from models/vexira.pt --out models/vexira_sft.pt
```

---

## Dizin yapısı

```
translate.py        ← ana kullanım (CLI + Python API + server)
glossary.py         sözlük: tam eşleşme, doğrulama, beyne gömme
evaluate.py         FLORES BLEU/chrF++ + terim tutarlılığı ölçümü
tokenizer.py        SentencePiece sarmalayıcı, özel token mantığı
config.py           mimari
model.py            Vexira (enc-dec, RMSNorm, SwiGLU, KV cache)
vexira.sh/.bat/.ps1 çift tıkla menü (terminal bilmeyene)
menu.py             menünün kendisi
webui.py            tarayıcı arayüzü (tek komut, bağımlılık yok)
examples/integrate.py  kendi projene bağlama (ENTEGRASYON ARAYÜZÜ)
examples/minimal.py modeli en az kodla çalıştırma (~40 satır, öğretici)
samples/            örnek girdi (TR + EN, kolaydan zora)
models/             ağırlıklar (Hugging Face'ten iner)
training/           eğitim tarafı — kullanmak için gerekmez
```

## Lisans

Apache-2.0. Bkz. [LICENSE](LICENSE).
