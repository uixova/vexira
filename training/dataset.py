# -*- coding: utf-8 -*-
"""
Vexira veri katmanı — memmap bitext + uzunluk-kovalı token batching.

Disk formatı (data/build_bin.py üretir):
  vexira.bin       uint16 düz token dizisi (kaynak ve hedef ardışık yazılır)
  vexira.idx.npy   int64 (N, 4) = [src_off, src_len, tgt_off, tgt_len]
  vexira.meta.json {vocab, n_examples, n_tokens, directions, spm_sha, ...}

Neden uint16: vocab 32000 < 65536 olduğu için token id'leri 2 bayta sığıyor,
disk ve RAM yarıya iniyor.

Neden PyTorch Dataset/DataLoader YOK: NMT'de batch'ler örnek sayısıyla değil
TOKEN bütçesiyle kurulur ve benzer uzunluklar bir araya gelmelidir. Rastgele
batch'te padding israfı %40'ı bulur; uzunluk kovalarıyla %5'e iner — CPU'da bu
doğrudan 1.5-2x hız demek.
"""

import json
import os

import numpy as np
import torch

def _total_ram():
    """Toplam fiziksel RAM (bayt). psutil yok — /proc/meminfo yeterli."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 16 * 1024 ** 3          # bilinmiyorsa temkinli varsayım


IGNORE_INDEX = -100     # chunked_ce bunu eler
PAD_ID, BOS_ID = 0, 2


# =========================== DİSK ===========================

class BitextStore:
    """Tokenize edilmiş bitext'e salt-okunur erişim."""

    def __init__(self, prefix, ram="auto", world_size=1, verbose=True):
        """prefix: '.../vexira'  ->  .bin / .idx.npy / .meta.json

        ram: "auto" (önerilen) | True (zorla) | False (hep memmap)

        ⚠️ RAM'e almak DDP'de rank BAŞINA maliyettir. 10.9 GB bin + 3.9 GB index
        iki rank'te ~29.6 GB eder ve OOM killer rank'i SIGKILL ile öldürür
        (gerçekte yaşandı). "auto" toplam maliyeti ölçüp bol marj varsa promote
        eder, yoksa memmap'te kalır — sayfa önbelleği zaten işi görür.
        """
        self.prefix = prefix
        self.meta = json.load(open(prefix + ".meta.json", encoding="utf-8"))

        path = prefix + ".bin"
        want = self.meta["n_tokens"] * 2
        got = os.path.getsize(path)
        if got != want:
            raise ValueError(
                f"{path} boyutu tutmuyor: {got} bayt, beklenen {want} "
                f"(n_tokens={self.meta['n_tokens']}). Veri bozuk ya da meta bayat.")

        idx_path = prefix + ".idx.npy"
        idx_bytes = os.path.getsize(idx_path)
        total_ram = _total_ram()
        # rank başına maliyet x rank sayısı, %45 tavan (optimizer/aktivasyon da yer ister)
        cost = (got + idx_bytes) * max(1, world_size)
        budget = 0.45 * total_ram

        if ram == "auto":
            use_ram = cost < budget
        else:
            use_ram = bool(ram)

        if verbose:
            print(f"[veri] bin {got/1e9:.2f} GB + idx {idx_bytes/1e9:.2f} GB "
                  f"x{world_size} rank = {cost/1e9:.1f} GB · RAM {total_ram/1e9:.0f} GB "
                  f"· {'RAM' if use_ram else 'memmap'}", flush=True)
            if ram is True and cost > budget:
                print(f"[veri] ⚠️  --ram-data 1 zorlandı ama {cost/1e9:.1f} GB > "
                      f"bütçe {budget/1e9:.1f} GB — OOM riski", flush=True)

        mm = None if use_ram else "r"
        self.index = np.load(idx_path, mmap_mode=mm)                # (N,4) int64
        if self.index.ndim != 2 or self.index.shape[1] != 4:
            raise ValueError(f"bozuk index şekli: {self.index.shape}, (N,4) bekleniyor")

        if use_ram:
            self.tokens = np.fromfile(path, dtype=np.uint16)
        else:
            self.tokens = np.memmap(path, dtype=np.uint16, mode="r")

        # Uzunlukları HER ZAMAN RAM'de küçük dizilerde tut.
        # Batch kurulumu uzunluklara rastgele erişir; bunu 3.9 GB'lık memmap
        # index üzerinden yapmak epoch başına milyonlarca sayfa hatası demek.
        # 121.7M x 2 x uint16 = 487 MB — memmap'ten okumaya kıyasla ucuz.
        if use_ram:
            self.src_len = self.index[:, 1].astype(np.uint16)
            self.tgt_len = self.index[:, 3].astype(np.uint16)
        else:
            n = len(self.index)
            self.src_len = np.empty(n, dtype=np.uint16)
            self.tgt_len = np.empty(n, dtype=np.uint16)
            step = 4_000_000                       # sıralı okuma, sayfa dostu
            for a in range(0, n, step):
                b = min(a + step, n)
                blk = np.asarray(self.index[a:b])
                self.src_len[a:b] = blk[:, 1]
                self.tgt_len[a:b] = blk[:, 3]
            if verbose:
                print(f"[veri] uzunluk dizileri RAM'de "
                      f"({2 * n * 2 / 1e6:.0f} MB)", flush=True)

    def __len__(self):
        return len(self.index)

    @property
    def vocab_size(self):
        return self.meta["vocab"]

    def get(self, i):
        so, sl, to, tl = self.index[i]
        return (self.tokens[so:so + sl].astype(np.int64),
                self.tokens[to:to + tl].astype(np.int64))

    def split(self, val_frac=0.002, val_max=20000, seed=42):
        """Deterministik train/val ayrımı. Val küçük tutulur — eğitim sırasında
        sık koşacak, asıl kalite ölçümü FLORES üzerinden evaluate.py ile yapılır."""
        n = len(self)
        rng = np.random.default_rng(seed)
        n_val = min(val_max, max(1, int(n * val_frac)))
        val = rng.choice(n, size=n_val, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[val] = False
        # train SIRALI kalır: batch üretici blok-yerel okuma yapabilsin diye.
        # Karıştırma LengthBucketBatcher içinde blok düzeyinde yapılıyor.
        train = np.flatnonzero(mask)
        return train, np.sort(val)


# =========================== BATCHING ===========================

class LengthBucketBatcher:
    """Token bütçeli, uzunluk kovalı batch üretici.

    Algoritma (standart NMT):
      1. örnek sırasını karıştır
      2. chunk_size'lık bloklara böl, HER BLOĞU uzunluğa göre sırala
      3. blok içinde token bütçesi dolana kadar batch topla
      4. batch sırasını karıştır (uzunluk sıralı gitmesin, gradyan biaslanmasın)
    Böylece hem rastgelelik korunur hem padding israfı düşer.
    """

    def __init__(self, store, indices, max_tokens=12000, max_sents=256,
                 chunk_size=20000, shuffle=True, seed=0, drop_last=False,
                 block_size=8192, n_mix=64):
        self.store = store
        self.indices = np.asarray(indices)
        self.max_tokens = max_tokens
        self.max_sents = max_sents
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.block_size = block_size    # ardışık okuma birimi (~0.7 MB token)
        self.n_mix = n_mix              # aynı anda harmanlanan uzak blok sayısı
        self._len = None

    def _lengths(self, idx):
        # batch maliyeti max(src_len, tgt_len) ile ölçeklenir; ikisinin büyüğüne göre kovala
        # RAM'deki uint16 dizileri kullan — memmap index'e rastgele erişme
        return np.maximum(self.store.src_len[idx], self.store.tgt_len[idx])

    def _mixed_chunks(self, rng):
        """Dosya-yerel ama domain-karışık parçalar üret.

        Tam rastgele permütasyon memmap'te örnek başına rastgele sayfa okuması
        demek: ölçüldü, 25 ms/batch (soğuk önbellekte 196 ms). Blok-yerel okuma
        4.2 ms/batch. Tek epoch koştuğumuz için her sayfa bir kez okunuyor,
        yani önbellek kurtarmıyor — erişim deseni belirleyici.

        Ama sadece ardışık blok almak da olmaz: .bin kaynak korpus sırasıyla
        yazılı (önce opensubs, sonra ccmatrix...), tek blok = tek domain =
        batch'ler homojen olur ve eğitim bozulur.

        Çözüm: n_mix adet UZAK bloğu birleştir. Okuma n_mix ardışık akış
        (yerel), içerik ise dosyanın her yerinden (karışık).
        """
        idx = self.indices
        n = len(idx)
        block = max(1, self.block_size)
        n_blk = (n + block - 1) // block
        order = rng.permutation(n_blk)
        for a in range(0, n_blk, self.n_mix):
            sel = order[a:a + self.n_mix]
            parts = [idx[b * block:(b + 1) * block] for b in sel]
            if not parts:
                continue
            merged = np.concatenate(parts)
            yield merged[rng.permutation(len(merged))]

    def batches(self, epoch=0):
        rng = np.random.default_rng(self.seed + epoch)
        if self.shuffle:
            chunks = self._mixed_chunks(rng)
        else:
            idx = self.indices
            chunks = (idx[i:i + self.chunk_size]
                      for i in range(0, len(idx), self.chunk_size))

        out = []
        for chunk in chunks:
            order = np.argsort(self._lengths(chunk), kind="stable")
            chunk = chunk[order]

            cur, cur_max = [], 0
            for i in chunk:
                L = int(max(self.store.src_len[i], self.store.tgt_len[i]))
                nxt_max = max(cur_max, L)
                # +1: decoder girdisi <bos> ile bir uzuyor
                if cur and ((len(cur) + 1) * (nxt_max + 1) > self.max_tokens
                            or len(cur) >= self.max_sents):
                    out.append(np.array(cur))
                    cur, cur_max = [i], L
                else:
                    cur.append(i)
                    cur_max = nxt_max
            if cur and not self.drop_last:
                out.append(np.array(cur))

        if self.shuffle:
            rng = np.random.default_rng(self.seed + 100000 + epoch)
            out = [out[j] for j in rng.permutation(len(out))]
        self._len = len(out)
        return out

    def __len__(self):
        if self._len is None:
            self._len = len(self.batches(0))
        return self._len


def collate(store, batch_idx, device=None, pin=False):
    """indeks dizisi -> (src, tgt_in, tgt_out, src_pad) tensörleri.

    tgt_in  = <bos> + hedef[:-1]     (teacher forcing kayması)
    tgt_out = hedef                  (son eleman </s>)
    Pad konumları tgt_out'ta IGNORE_INDEX, böylece kayba girmez.
    """
    B = len(batch_idx)
    sl = store.src_len[batch_idx]
    tl = store.tgt_len[batch_idx]
    S, T = int(sl.max()), int(tl.max())

    src = np.full((B, S), PAD_ID, dtype=np.int64)
    tgt_in = np.full((B, T), PAD_ID, dtype=np.int64)
    tgt_out = np.full((B, T), IGNORE_INDEX, dtype=np.int64)

    for r, i in enumerate(batch_idx):
        s, t = store.get(i)
        src[r, :len(s)] = s
        tgt_in[r, 0] = BOS_ID
        tgt_in[r, 1:len(t)] = t[:-1]
        tgt_out[r, :len(t)] = t

    src = torch.from_numpy(src)
    tgt_in = torch.from_numpy(tgt_in)
    tgt_out = torch.from_numpy(tgt_out)
    src_pad = src != PAD_ID                     # True = geçerli token

    if pin:
        src, tgt_in, tgt_out, src_pad = (x.pin_memory() for x in
                                         (src, tgt_in, tgt_out, src_pad))
    if device is not None:
        nb = pin
        src = src.to(device, non_blocking=nb)
        tgt_in = tgt_in.to(device, non_blocking=nb)
        tgt_out = tgt_out.to(device, non_blocking=nb)
        src_pad = src_pad.to(device, non_blocking=nb)
    return src, tgt_in, tgt_out, src_pad


def prefetch(store, batches, device=None, pin=False, depth=6):
    """collate'i TEK arka plan thread'inde yapıp GPU beklemesin diye önden hazırla.

    Tek worker bilinçli: çok worker + sıra tamponu denendi, ek yükü (queue +
    dict yeniden sıralama) I/O beklemesinden büyük çıktı ve ÖLÇÜMDE YAVAŞLATTI
    (0.6x). Tek worker'da sıra doğal korunur, ek yük ~0.1 ms.

    collate zamanının çoğu memmap okumasıdır (GIL bırakılır), o yüzden thread
    gerçekten örtüşür. Kazanç I/O gecikmesiyle orantılı: sayfa önbellekte ise
    kazanç yok, diskten okunuyorsa belirgin.
    """
    import queue
    import threading

    q = queue.Queue(maxsize=depth)
    _DONE = object()

    def worker():
        try:
            for b in batches:
                q.put(collate(store, b, device=None, pin=pin))
        except Exception as e:                                   # noqa: BLE001
            q.put(e)
        finally:
            q.put(_DONE)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is _DONE:
            return
        if isinstance(item, Exception):
            raise item
        if device is not None:
            item = tuple(x.to(device, non_blocking=pin) for x in item)
        yield item


def shard_batches(batches, rank, world_size):
    """DDP: her rank batch listesinin kendi dilimini alır. Tüm rank'ler AYNI
    sayıda batch görmeli, yoksa all_reduce'ta kilitlenme olur -> kırpıyoruz."""
    if world_size <= 1:
        return batches
    n = (len(batches) // world_size) * world_size
    return batches[rank:n:world_size]


# =========================== SENTETİK (test/bench) ===========================

def make_synthetic(path_prefix, n=2000, vocab=32000, src_len=(8, 40), tgt_len=(8, 40),
                   seed=0):
    """bench.py ve boru hattı testi için sahte bitext üret. Gerçek veri gerekmez."""
    rng = np.random.default_rng(seed)
    idx, buf, off = [], [], 0
    for _ in range(n):
        sl = int(rng.integers(*src_len))
        tl = int(rng.integers(*tgt_len))
        s = rng.integers(20, vocab, size=sl, dtype=np.uint16)
        s[0] = 4                                  # <2tr>
        s[-1] = 3                                 # </s>
        t = rng.integers(20, vocab, size=tl, dtype=np.uint16)
        t[-1] = 3
        idx.append((off, sl, off + sl, tl))
        buf.append(s)
        buf.append(t)
        off += sl + tl
    tokens = np.concatenate(buf)
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
    tokens.tofile(path_prefix + ".bin")
    np.save(path_prefix + ".idx.npy", np.array(idx, dtype=np.int64))
    json.dump({"vocab": vocab, "n_examples": n, "n_tokens": int(len(tokens)),
               "directions": {"synthetic": n}, "spm_sha": "synthetic"},
              open(path_prefix + ".meta.json", "w"), ensure_ascii=False, indent=2)
    return path_prefix


def _test():
    import tempfile
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("  ✓ " if cond else "  ✗ ") + name + (f"  [{extra}]" if extra else ""))
        ok = ok and cond

    with tempfile.TemporaryDirectory() as d:
        p = make_synthetic(os.path.join(d, "vx"), n=3000)
        store = BitextStore(p)
        check("store yüklendi", len(store) == 3000, str(len(store)))

        tr, va = store.split(val_frac=0.01)
        check("train/val ayrık", len(set(tr) & set(va)) == 0)
        check("val boyutu", len(va) == 30, str(len(va)))

        b = LengthBucketBatcher(store, tr, max_tokens=4000, seed=1)
        batches = b.batches(0)
        check("tüm örnekler tam bir kez", sum(len(x) for x in batches) == len(tr),
              f"{sum(len(x) for x in batches)} vs {len(tr)}")

        # padding verimliliği: kovalı vs rastgele
        def waste(bs):
            used = tot = 0
            for bb in bs:
                L = int(max(store.index[bb, 3]))
                used += int(store.index[bb, 3].sum())
                tot += len(bb) * L
            return 1 - used / tot
        rnd = np.random.default_rng(0).permutation(tr)
        rnd_b = [rnd[i:i + 64] for i in range(0, len(rnd), 64)]
        w_buck, w_rand = waste(batches), waste(rnd_b)
        check("kovalı padding israfı rastgeleden düşük",
              w_buck < w_rand, f"kovalı={w_buck*100:.1f}% rastgele={w_rand*100:.1f}%")

        src, ti, to, sp = collate(store, batches[0])
        check("collate şekilleri", src.shape[0] == ti.shape[0] == to.shape[0])
        check("tgt_in <bos> ile başlıyor", bool((ti[:, 0] == BOS_ID).all()))
        check("src_pad maskesi doğru", bool((sp == (src != PAD_ID)).all()))

        # teacher forcing kayması: tgt_in[1:] == tgt_out[:-1] (pad olmayan bölgede)
        row = 0
        n_real = int((to[row] != IGNORE_INDEX).sum())
        shift_ok = bool((ti[row, 1:n_real] == to[row, :n_real - 1]).all())
        check("teacher forcing kayması doğru", shift_ok)
        # tgt_in'de pad olan her konum tgt_out'ta IGNORE_INDEX olmalı ve tersi
        pad_pos = ti == PAD_ID
        pad_pos[:, 0] = False                      # ilk konum her zaman <bos>
        check("pad hedefler IGNORE_INDEX",
              bool((to[pad_pos] == IGNORE_INDEX).all()) and int((to == IGNORE_INDEX).sum()) > 0,
              f"{int((to == IGNORE_INDEX).sum())} maskeli konum")

        s0 = shard_batches(batches, 0, 2)
        s1 = shard_batches(batches, 1, 2)
        check("DDP shard eşit uzunlukta", len(s0) == len(s1), f"{len(s0)} vs {len(s1)}")
        check("DDP shard'ları çakışmıyor",
              len({id(x) for x in s0} & {id(x) for x in s1}) == 0)

    print("\n" + ("HEPSİ GEÇTİ ✓" if ok else "BAŞARISIZ ✗"))
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        raise SystemExit(0 if _test() else 1)
    print("kullanım: python dataset.py test")
