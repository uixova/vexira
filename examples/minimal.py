# -*- coding: utf-8 -*-
"""
Vexira — EN AZ KOD ile çeviri.  (~40 satır, tek bağımlılık: torch + sentencepiece)

translate.py 650+ satır çünkü tam bir ARAÇ: sözlük, .srt, sunucu modu, uzun
girdi bölme, çıktı onarımı, regresyon testi. Modeli çalıştırmak için o kadarı
GEREKMİYOR. Aşağısı çekirdeğin tamamı.

    python examples/minimal.py "Hello world"

Buradan eksik olanlar (translate.py'de var):
  - sözlük (terim tutarlılığı)          - 128 token üstü girdiyi bölme
  - yer tutucu / kaçış dizisi onarımı   - beam search (burada greedy)
  - toplu çeviri, uzunluk kovalama      - .srt, sunucu modu

Farkın somut hâli — aynı cümle, iki yol:

    minimal.py     "The renderer failed to start." -> "Tezgah başlatılamadı."
    translate.py   (sözlük + --domain ui)          -> "Oluşturucu başlatılamadı."

"Tezgah" yanlış. Model kendi başına bağlamsız bir teknik terimi ıskalayabiliyor;
sözlük katmanı tam olarak bunun içindi. Yani bu dosya "modeli nasıl çağırırım"
sorusunun cevabı, "en iyi çeviriyi nasıl alırım" sorusunun değil.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VexiraConfig          # noqa: E402
from model import Vexira                 # noqa: E402
from tokenizer import VexiraTokenizer    # noqa: E402

CKPT = "models/vexira_sft.pt"
SPM = "models/vexira_spm.model"


def load(ckpt=CKPT, spm=SPM):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Çalışma ayarı modelin İÇİNDE; thread sayısını oradan al (fazlası yavaşlatır).
    torch.set_num_threads((ck.get("runtime") or {}).get("threads", 4))
    cfg = VexiraConfig.from_dict(ck["config"])
    cfg.dropout = 0.0
    model = Vexira(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, VexiraTokenizer(spm), cfg


@torch.no_grad()
def translate(model, tok, cfg, text, to="tr", domain="doc", max_new=128):
    """Greedy decode. Tek cümle için beam ile fark küçük.

    domain="doc" varsayılan: serbest metin için doğrusu bu. "sub" altyazı
    DOSYASI içindir; orada model kısa selamlamayı iki konuşmacılı diyaloğa
    çeviriyor ("Good morning." -> "Günaydın. - Günaydın."), çünkü OpenSubtitles
    verisinde iki replik tek satırda geçiyor. translate.py bunu ayrıca
    onarıyor; burada doğru alanı seçmek yetiyor.
    """
    src = torch.tensor([tok.encode_source(text, to, domain=domain,
                                          max_len=128)])
    mem = model.encode(src)                      # encoder BİR KEZ koşar
    caches = model.init_cache(mem, max_len=cfg.max_pos)   # KV cache
    out = [tok.bos_id]
    for pos in range(max_new):
        logits = model.decode_step(torch.tensor([[out[-1]]]), caches, pos)
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id:
            break
        out.append(nxt)
    return tok.decode(out[1:])


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Hello world"
    model, tok, cfg = load()
    print(translate(model, tok, cfg, text, to="tr"))
