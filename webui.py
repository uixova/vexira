# -*- coding: utf-8 -*-
"""
Vexira web arayüzü — tek komut, tarayıcıda açılır.

    python webui.py            # http://127.0.0.1:8770 açılır

YENİ BAĞIMLILIK YOK. Flask/FastAPI yerine stdlib http.server kullanılıyor:
projenin tamamı hâlâ torch + sentencepiece. Arayüz tek dosyada gömülü HTML,
harici CSS/JS/font çekmiyor — internetsiz makinede de çalışır.

Sadece 127.0.0.1'e bağlanır; dışarıya açık DEĞİLDİR. Ağa açmak istersen
--host 0.0.0.0 ver, ama o zaman kimlik doğrulaması olmadığını unutma.
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from translate import Translator, DEFAULT_CKPT, resolve_runtime  # noqa: E402

PAGE = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vexira — TR ↔ EN</title><style>
:root{--bg:#0f1115;--panel:#171a21;--line:#272b35;--fg:#e6e8ee;--dim:#8b93a7;
--acc:#5b8cff;--ok:#3ddc97}
@media(prefers-color-scheme:light){:root{--bg:#f6f7f9;--panel:#fff;--line:#e2e5ea;
--fg:#14161a;--dim:#6b7280;--acc:#2563eb}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;
background:var(--bg);color:var(--fg)}
.wrap{max-width:1000px;margin:0 auto;padding:24px 18px 60px}
h1{font-size:20px;margin:0 0 2px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin-bottom:18px}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
select,button{font:inherit;background:var(--panel);color:var(--fg);
border:1px solid var(--line);border-radius:8px;padding:7px 11px}
button{cursor:pointer}
button.go{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600;
padding:7px 18px}
button.go:disabled{opacity:.55;cursor:default}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.cell{display:flex;flex-direction:column}
.lbl{font-size:12px;color:var(--dim);margin-bottom:5px;display:flex;
justify-content:space-between;align-items:center}
textarea{width:100%;min-height:300px;resize:vertical;padding:12px;
background:var(--panel);color:var(--fg);border:1px solid var(--line);
border-radius:10px;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
textarea[readonly]{background:color-mix(in srgb,var(--panel) 92%,var(--acc) 8%)}
.mini{background:none;border:1px solid var(--line);padding:3px 9px;font-size:12px;
border-radius:6px;color:var(--dim)}
.stat{margin-top:10px;font-size:12.5px;color:var(--dim);min-height:18px}
.stat b{color:var(--ok);font-weight:600}
kbd{background:var(--panel);border:1px solid var(--line);border-bottom-width:2px;
border-radius:5px;padding:1px 5px;font-size:11px;font-family:ui-monospace}
</style></head><body><div class="wrap">
<h1>Vexira</h1>
<div class="sub">80.8M · Türkçe ↔ İngilizce · tamamen yerel, GPU yok</div>

<div class="bar">
  <select id="dir">
    <option value="tr">EN → TR</option>
    <option value="en">TR → EN</option>
  </select>
  <select id="dom">
    <option value="sub">altyazı</option>
    <option value="doc">düzyazı</option>
    <option value="ui">arayüz</option>
    <option value="ocr">ekran metni</option>
  </select>
  <select id="beam">
    <option value="2">beam 2 (hızlı)</option>
    <option value="4">beam 4 (kaliteli)</option>
    <option value="1">beam 1 (en hızlı)</option>
  </select>
  <button class="go" id="go">Çevir</button>
  <span style="color:var(--dim);font-size:12px">
    <kbd>Ctrl</kbd>+<kbd>Enter</kbd></span>
</div>

<div class="grid">
  <div class="cell">
    <div class="lbl"><span>Kaynak — her satır ayrı çevrilir</span>
      <button class="mini" id="clr">temizle</button></div>
    <textarea id="src" placeholder="Metni buraya yaz…
Her satır bağımsız çevrilir.
128 token üstü satır cümleden bölünür."></textarea>
  </div>
  <div class="cell">
    <div class="lbl"><span>Çeviri</span>
      <button class="mini" id="cp">kopyala</button></div>
    <textarea id="dst" readonly placeholder="Sonuç burada…"></textarea>
  </div>
</div>
<div class="stat" id="stat"></div>
</div><script>
const $=i=>document.getElementById(i);
async function run(){
  const text=$('src').value; if(!text.trim())return;
  $('go').disabled=true; $('stat').textContent='çevriliyor…';
  const t0=performance.now();
  try{
    const r=await fetch('/api/translate',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({lines:text.split('\\n'),to:$('dir').value,
                           domain:$('dom').value,beam:+$('beam').value})});
    const j=await r.json();
    if(!j.ok){$('stat').textContent='hata: '+j.error;return;}
    $('dst').value=j.out.join('\\n');
    const ms=performance.now()-t0, n=j.out.filter(x=>x.trim()).length;
    $('stat').innerHTML=`<b>${n}</b> satır · <b>${ms.toFixed(0)} ms</b> `
      +`(${(ms/Math.max(1,n)).toFixed(0)} ms/satır) · sözlükten `
      +`<b>${j.stats.exact}</b> tam eşleşme · <b>${j.stats.model}</b> modelden`
      +(j.stats.split?` · ${j.stats.split} uzun satır bölündü`:'');
  }catch(e){$('stat').textContent='hata: '+e}
  finally{$('go').disabled=false}
}
$('go').onclick=run;
$('clr').onclick=()=>{$('src').value='';$('dst').value='';$('stat').textContent=''};
$('cp').onclick=()=>{navigator.clipboard.writeText($('dst').value);
  $('stat').textContent='panoya kopyalandı'};
document.addEventListener('keydown',e=>{
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter')run()});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    engine = None
    lock = threading.Lock()

    def log_message(self, *a):
        pass                              # istek başına log basma

    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, "yok", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/translate":
            return self._send(404, "{}", "application/json")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
            lines = [l for l in (req.get("lines") or [])]
            # Boş satırlar modele gitmez ama SIRA korunur: çıktı satır satır
            # kaynağın karşısına gelsin diye yerlerine boş string konuyor.
            idx = [i for i, l in enumerate(lines) if l.strip()]
            with self.lock:                      # model tek örnek, seri kullan
                eng = Handler.engine
                before = dict(eng.stats)
                got = eng.translate([lines[i] for i in idx],
                                    tgt_lang=req.get("to", "tr"),
                                    domain=req.get("domain", "sub"),
                                    beam=int(req.get("beam", 2)))
                delta = {k: eng.stats[k] - before.get(k, 0) for k in eng.stats}
            out = [""] * len(lines)
            for i, g in zip(idx, got):
                out[i] = g
            self._send(200, json.dumps({"ok": True, "out": out, "stats": delta},
                                       ensure_ascii=False), "application/json")
        except Exception as e:                                    # noqa: BLE001
            self._send(200, json.dumps({"ok": False,
                                        "error": f"{type(e).__name__}: {e}"}),
                       "application/json")


def main():
    ap = argparse.ArgumentParser(description="Vexira web arayüzü")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--spm", default=None)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 ağa açar — kimlik doğrulaması YOK")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    rt = {}
    if os.path.exists(args.ckpt):
        try:
            rt = torch.load(args.ckpt, map_location="cpu", weights_only=False,
                            mmap=True).get("runtime") or {}
        except Exception:                                         # noqa: BLE001
            rt = {}
    torch.set_num_threads(resolve_runtime(rt, threads=args.threads)["threads"])

    print(f"model yükleniyor: {args.ckpt}", flush=True)
    Handler.engine = Translator(args.ckpt, args.spm, "cpu")
    i = Handler.engine.info
    print(f"hazır — {i['params']/1e6:.1f}M param, sözlük {i['glossary']} terim, "
          f"{i['runtime']['threads']} thread")

    url = f"http://{args.host}:{args.port}"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"arayüz: {url}   (durdurmak için Ctrl+C)")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nkapatılıyor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
