# Vexira

**An 80.8M-parameter TR↔EN translation model.** Runs fully offline — no
internet, no API key, no account. Built for subtitles, game text, UI strings
and file content.

[Türkçe](README.md) · [Hugging Face](https://huggingface.co/uixova/vexira) · [GitHub](https://github.com/uixova/vexira)

| | |
|---|---|
| Parameters | 80.8M (fp16, ~490 MB) |
| Languages | Turkish ↔ English, bidirectional, single model |
| Speed | **78 ms/line** (12.9 lines/s) — 4 CPU threads, no GPU |
| FLORES-200 | en→tr **BLEU 29.1 / chrF++ 57.7** · tr→en **BLEU 35.2 / chrF++ 60.1** |
| Training | 7.56B tokens (4.67× Chinchilla) + UI-focused fine-tune |
| License | Apache-2.0 |

---

## Quick start

```bash
git clone https://github.com/uixova/vexira && cd vexira
pip install torch sentencepiece

# Download weights (~490 MB)
huggingface-cli download uixova/vexira vexira_sft.pt vexira_spm.model --local-dir models/
```

Then one line:

```bash
# Text
python translate.py --text "Hello world" --to tr
# -> Merhaba dünya

# File
python translate.py --file samples/sample_en.txt --to tr --domain doc

# Subtitles (.srt timestamps preserved)
python translate.py --srt movie.srt --to tr --out movie.tr.srt

# Persistent server — for OCR/TTS/STT layers, model loads once
python translate.py --server
```

**Browser UI** — one command, no new dependency (stdlib `http.server`):

```bash
python webui.py            # opens http://127.0.0.1:8770 automatically
```

Type on the left, copy from the right. Direction/domain/beam selectors,
`Ctrl+Enter` shortcut, live stats. The HTML is embedded in the file — it works
on a machine with no internet, and binds to `127.0.0.1` only.

From Python:

```python
from translate import Translator

tr = Translator()                      # finds models/vexira_sft.pt automatically
print(tr.translate(["Save", "Are you sure you want to quit?"],
                   tgt_lang="tr", domain="ui"))
# ['Kaydet', 'Çıkmak istediğinizden emin misiniz?']
```

### "Do I really need all that code?" — no

`translate.py` is 650+ lines because it is a full **tool**: glossary, `.srt`,
server mode, long-input splitting, output repair. None of that is required to
**run** the model. [`examples/minimal.py`](examples/minimal.py) is the entire
core — 40 lines:

```python
ck = torch.load("models/vexira_sft.pt", map_location="cpu", weights_only=False)
torch.set_num_threads(ck["runtime"]["threads"])        # settings live in the file
cfg = VexiraConfig.from_dict(ck["config"]); cfg.dropout = 0.0
model = Vexira(cfg); model.load_state_dict(ck["model"]); model.eval()

src    = torch.tensor([tok.encode_source(text, "tr", max_len=128)])
mem    = model.encode(src)                             # encoder runs once
caches = model.init_cache(mem, max_len=cfg.max_pos)    # KV cache
# ... greedy loop ...
```

```bash
python examples/minimal.py "Hello world, this is a small translation model."
# -> Merhaba dünya, bu küçük bir çeviri modeli.
```

The same sentence through both paths shows what you trade away:

```
minimal.py    "The renderer failed to start." -> "Tezgah başlatılamadı."      ✗
translate.py  (glossary + --domain ui)        -> "Oluşturucu başlatılamadı."  ✓
```

So the minimal file answers "how do I call the model"; the glossary layer is
what delivers the quality.

### Runtime settings ship inside the model

The right thread/beam values are model-specific and were found by measurement —
a user cannot be expected to know them, so they are **embedded in the
checkpoint**:

```python
ck["runtime"]   # {'threads': 4, 'beam': 2, 'batch_size': 32, 'domain': 'sub'}
```

Wherever the model goes, its settings go with it. Override order:

```
CLI flag     >  env var           >  embedded       >  code default
--threads 8     VEXIRA_THREADS=8     ck["runtime"]     RUNTIME_DEFAULTS
```

```bash
VEXIRA_THREADS=2 VEXIRA_BEAM=4 python translate.py --text "..." --to tr
```

### Try it with the bundled samples

`samples/` holds two files, ordered from easy to hard, so you can see exactly
where the model struggles:

```bash
python translate.py --file samples/sample_en.txt --to tr --domain doc   # EN -> TR
python translate.py --file samples/sample_tr.txt --to en --domain doc   # TR -> EN
```

They cover short UI labels, text with placeholders, subtitle lines, polysemous
words, prose, technical register, a paragraph above 128 tokens, and inverted or
idiomatic constructions.

### Key flags

| flag | default | note |
|---|---|---|
| `--to` | `tr` | target language: `tr` / `en` |
| `--domain` | `sub` | `sub` subtitles · `ui` interface · `doc` prose · `ocr` screen text |
| `--beam` | 2 | 4 gives about +0.5 BLEU but is ~1.7× slower |
| `--threads` | auto (4) | **do not raise** — more threads are slower at this size, see below |
| `--no-glossary` | off | disable the glossary (raw model output) |

`--domain` genuinely matters: with `ui` the glossary kicks in and `Save` →
`Kaydet` becomes deterministic; with `doc` the model decides for itself.

---

## Benchmarks

### FLORES-200 devtest (1012 sentences, beam 4, glossary off)

| model | en→tr BLEU | chrF++ | tr→en BLEU | chrF++ |
|---|---|---|---|---|
| `vexira.pt` (pretrained) | **29.53** | 57.84 | 35.03 | 59.98 |
| **`vexira_sft.pt`** (main) | 29.13 | 57.68 | **35.16** | **60.11** |

For reference on the same test set: NLLB-200 distilled (600M) scores roughly
26–28 en→tr, opus-mt-tc-big-en-tr (230M) around 30. Vexira sits in that band
with 80.8M parameters.

> tr→en looking higher is a metric artifact, not a model difference. BLEU
> matches surface words, and Turkish inflection makes `evden`/`eve` count as
> completely wrong. The chrF++ gap (60.1 vs 57.7) reflects reality better.

### Term consistency — a real 559-line Ren'Py file

The defect BLEU cannot see: the same term translated differently across lines.

| model | inconsistent terms | with glossary |
|---|---|---|
| `vexira.pt` | 33 | 29 |
| **`vexira_sft.pt`** | **19** | 21 |

This is where the fine-tune pays off: a **42% reduction**. On top of that the
glossary resolves **42%** of the 559 strings by exact match — the model is never
called for those, so inconsistency is mathematically impossible.

### Speed (12-core laptop, CPU only, no GPU)

| threads | beam 1 | beam 2 | beam 4 |
|---|---|---|---|
| 2 | 76.8 ms | 105.8 ms | 162.3 ms |
| **4** | 82.5 ms | **77.6 ms** | 135.5 ms |
| 8 | 86.7 ms | 133.5 ms | 181.6 ms |
| 12 | 338.9 ms | 466.0 ms | 367.7 ms |

**More threads make it slower.** At this size the per-layer work is small and
synchronization overhead dominates. That is why `translate.py` does not leave
the thread count to torch but caps it at 4 — a 6× difference out of the box.

---

## How it works

### Architecture

| | value |
|---|---|
| type | encoder-decoder (not decoder-only — far better parameter efficiency for translation) |
| encoder / decoder | 12 / 6 |
| d_model / d_ff / heads | 512 / 1408 / 8 |
| vocab | 32000 shared TR+EN SentencePiece **unigram** |
| **effective context** | **128 tokens** (~80–100 words) — [details](#effective-context-128-tokens) |
| positions | learned absolute (not RoPE) |
| norm / ffn | RMSNorm (pre-LN) / SwiGLU |
| embedding | tied (encoder + decoder + lm_head share one table) |
| total | **80.76M** — embed 16.4M · enc 38.5M · dec 25.6M |

**Why encoder-decoder.** Translation means seeing the source in full, then
producing the target. Matching this quality with a decoder-only stack costs
several times the parameters.

**Why 12+6.** The encoder dominates end-to-end latency: 12 layers over 48 source
tokens in one pass, versus 6 decoder layers over ~20 steps. Going 4→6 in the
decoder adds only 6% total cost; moving two layers from encoder to decoder makes
the model **5% faster** while strengthening Turkish inflection.

**Where meaning lives.** The embedding table (16.4M) is a context-free lookup —
meaning is not there. `bank` is one row; after 12 encoder layers "river bank"
and "bank account" are different vectors:

```
He works at the bank.               -> Bankada çalışıyor.
I went to the river bank to think.  -> Düşünmek için nehir kıyısına gittim.
She plays the piano every evening.  -> Her akşam piyano çalıyor.
The children play in the garden.    -> Çocuklar bahçede oynuyor.
```

Comparable classic NMT models (~77M) burn 43% of their budget on embeddings
because their vocab is 58–65k. In Vexira that share is 21% — roughly 17M more
parameters doing actual work.

**No quantization.** int8/ternary are deliberately absent: at this scale the
quality loss is not worth it. Speed comes from KV cache, computing encoder K/V
once, and the thread cap.

### Effective context: 128 tokens

`config.max_pos = 512`, but the **usable context is 128 tokens**. Pretraining
ran with `max_len=128`, so positional embeddings 128–511 were never updated:

```
pos   0-127 : norm 9.5 – 10.4     trained
pos 128-511 : norm 0.34           still at init
```

`Translator` **measures** this ceiling instead of trusting the config. Longer
input is split at sentence boundaries, the parts are chained with `<ctx>`, and
the outputs are joined — the same thing Google does. Measured: 16 sentences at
144 tokens preserved **16/16 concepts**.

Raising it to 512 is not worthwhile: only **0.31%** of the clean corpus exceeds
128 tokens. Parallel corpora are sentence-aligned by construction; Marian and
OPUS-MT are sentence-level too. A half-trained position band is worse than a
clean boundary.

### Term consistency — the glossary layer

A neural net translates every sentence from scratch and has no term memory.
That is a **consistency** problem, not a quality one, and more training data
does not fix it — Google and DeepL do not fix it in the model either, they add
a glossary/termbase layer.

**The glossary lives inside the weights.** `vexira_sft.pt` = weights + config +
step + **1,234 terms** + tokenizer. Nothing extra to carry around:

```bash
python glossary.py show models/vexira_sft.pt      # inspect
python glossary.py export models/vexira_sft.pt --out terms.tsv
python glossary.py build terms.tsv --out models/vexira_sft.pt --spm models/vexira_spm.model
```

Terms come from two independent sources: system locale catalogs
(`/usr/share/locale`, **human translations**) and domain terms. A term enters
the glossary only if it appears in **≥3 separate packages** and **≥70% of them
agree** on the same rendering — the decision is the de-facto consensus of
independent translators, not one person's opinion. Without consensus the term
is left out and the encoder resolves it from context.

Brand names are never translated:

```
Windows -> Windows (untranslated)     window -> pencere
```

Casing is carried from the input: `SAVE`→`KAYDET`, `Save`→`Kaydet`,
`save`→`kaydet`.

### Output repair

Two defects measured on the real Ren'Py file are repaired deterministically at
inference:

| defect | example | before | after |
|---|---|---|---|
| escape sequence split | `\n` → `\ n` | 14 | **0** |
| placeholder translated | `[text]` → `[metin]` | 58 | **10** |

Disable with `--no-repair`.

---

## Known limits

- Input above **128 tokens** is split and rejoined, not processed in one pass.
- **TR↔EN only.** No other languages.
- **Not a knowledge model.** It does not answer questions, chat, or reason.
- Poetry, wordplay, heavy idiom and slang are weak.
- The `<ocr>`, `<stt>`, `<tts>` domain tokens are reserved but **untrained** —
  they will be fed by a later fine-tune.
- The glossary targets UI terminology; with `--domain doc` the model decides
  for itself in prose.

---

## Training

| stage | data | result |
|---|---|---|
| Pretraining | 60.9M pairs → 121.7M examples (both directions), 5.46B tokens/epoch | 4 sessions, 7.56B tokens (4.67× Chinchilla), val 2.6444 |
| Fine-tune | 133,753 examples, 3 epochs, LR 5e-5 | val 2.4147, 12 minutes |

**Data sources:** 23 OPUS corpora (OpenSubtitles, CCMatrix, HPLT, CCAligned,
WikiMatrix, TED, Tatoeba, KDE4, SETIMES …) plus system locale catalogs.
FLORES-200 devtest was **never** part of training.

**Epoch policy:** two epochs maximum. Volume came from fresh sources rather than
repetition — showing a small model the same data again is memorization.

The fine-tune set is deliberately balanced: 51% UI / 49% general, with a length
distribution matching pretraining (mean 19.4 vs 19.4 tokens, ≥30 tokens 17.5% vs
18.1%). The goal was to teach UI terminology without losing general translation,
and the benchmarks confirm it held.

Training code lives under `training/`:

```bash
python training/train_spm.py                    # tokenizer
python training/build_bin.py                    # tokenize -> .bin
python training/train.py --data ... --preset main
python training/train.py --data ... --finetune --init-from models/vexira.pt --out models/vexira_sft.pt
```

---

## Layout

```
translate.py        ← main entry point (CLI + Python API + server)
glossary.py         glossary: exact match, validation, embedding into weights
evaluate.py         FLORES BLEU/chrF++ and term-consistency measurement
tokenizer.py        SentencePiece wrapper, special-token logic
config.py           architecture
model.py            Vexira (enc-dec, RMSNorm, SwiGLU, KV cache)
webui.py            browser UI (one command, no dependency)
examples/minimal.py run the model with the least possible code (~40 lines)
samples/            sample input (TR + EN, easy to hard)
models/             weights (downloaded from Hugging Face)
training/           training side — not needed to use the model
```

## License

Apache-2.0. See [LICENSE](LICENSE).
