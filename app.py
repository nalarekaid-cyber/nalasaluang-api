"""
NALASALUANG Backend v3.1 — Cloud Edition (RAM Optimized)
"""
import os, random, time, json, gc
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download

HF_DATASET = "Nalareka/nalasaluang-frasa"
DATA_DIR = "/tmp/nalasaluang_data"
OUTPUT_DIR = "/tmp/nalasaluang_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CACHE_SIZE = 200
_frasa_cache = {}

INSTRUMEN = {
    'tigo': {
        'kode': 'tigo', 'nama': 'Patiak Tigo',
        'nada': [249.0, 271.5, 309.8, 332.2, 384.0, 408.8, 503.2, 546.0],
        'nama_nada': ['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'],
        'mode': 'raw_files', 'folder': 'patiak_tigo_final',
        'files': {
            'PEMBUKAAN': '01_PEMBUKAAN_final.wav',
            'PENUTUPAN': '02_PENUTUPAN_final.wav',
            'DUODUO':    '03_ISI_BIASA_DuoDuo_final.wav',
            'GURAU':     '04_ISI_GURAU_UrangBasiang_final.wav',
            'SEDIH':     '05_ISI_SEDIH_RaakImah_final.wav',
        },
        'pool_json': None, 'fmin': 200, 'fmax': 600,
    },
    'ampek': {
        'kode': 'ampek', 'nama': 'Patiak Ampek',
        'nada': [229.0, 241.9, 262.1, 309.9, 460.5, 629.6],
        'nama_nada': ['A1', 'A2', 'A3', 'A4', 'P1', 'P2'],
        'mode': 'frasa', 'folder': 'patiak_ampek_frasa',
        'files': None, 'pool_json': 'pakem_patiak_ampek_pool.json',
        'fmin': 180, 'fmax': 700,
    },
    'bansi': {
        'kode': 'bansi', 'nama': 'Bansi',
        'nada': [479.0, 534.7, 598.3, 630.1, 658.0, 701.7, 793.2, 844.9, 888.7, 952.3, 1067.7],
        'nama_nada': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'],
        'mode': 'frasa', 'folder': 'bansi_frasa',
        'files': None, 'pool_json': 'bansi_pool.json',
        'fmin': 400, 'fmax': 1200,
    },
}

print("=" * 60)
print("NALASALUANG Backend v3.1 — RAM Optimized")
print("=" * 60)
print(f"\n📥 Download dataset: {HF_DATASET}")
t0 = time.time()
try:
    snapshot_download(repo_id=HF_DATASET, repo_type="dataset",
                      local_dir=DATA_DIR, local_dir_use_symlinks=False,
                      allow_patterns=["*.json", "*.wav"])
    print(f"✅ Selesai dalam {time.time()-t0:.1f}s")
except Exception as e:
    print(f"⚠️  {e}")


def load_frasa(path):
    if path in _frasa_cache:
        return _frasa_cache[path]
    if not os.path.exists(path):
        return None
    try:
        y, sr = sf.read(path, dtype='float32')
        if len(y.shape) > 1: y = y.mean(axis=1)
        y_int16 = (y * 32767).astype(np.int16)
        if len(_frasa_cache) >= CACHE_SIZE:
            oldest = next(iter(_frasa_cache))
            del _frasa_cache[oldest]
        _frasa_cache[path] = (y_int16, sr)
        return _frasa_cache[path]
    except Exception as e:
        print(f"     ⚠️ {path}: {e}")
        return None


def int16_to_float(y_int16):
    return y_int16.astype(np.float32) / 32767.0


def detect_phrases_from_file(y, sr, top_db=35, min_silence=0.3, min_phrase=1.5):
    frame = 1024; hop = 512
    n = (len(y) - frame) // hop + 1
    if n <= 0: return []
    rms = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p = i * hop
        rms[i] = np.sqrt(np.mean(y[p:p+frame]**2))
    mx = float(np.max(rms))
    if mx < 1e-6: return []
    thr = mx * (10 ** (-top_db / 20.0))
    is_sp = rms > thr
    phrases, in_p, st = [], False, 0.0
    for i, s in enumerate(is_sp):
        t = i * hop / sr
        if s and not in_p: st = t; in_p = True
        elif not s and in_p:
            if t - st >= min_phrase: phrases.append((int(st*sr), int(t*sr)))
            in_p = False
    if in_p and len(y)/sr - st >= min_phrase:
        phrases.append((int(st*sr), len(y)))
    merged = []
    for s, e in phrases:
        if merged and (s - merged[-1][1])/sr < min_silence:
            merged[-1] = (merged[-1][0], e)
        else: merged.append((s, e))
    return merged


def detect_pitch_fft(y, sr, fmin, fmax):
    ws = 4096
    if len(y) < ws: return []
    positions = np.linspace(0, len(y) - ws, min(20, max(1, len(y)//ws)), dtype=int)
    freqs = np.fft.rfftfreq(ws, 1.0/sr)
    fm = (freqs >= fmin) & (freqs <= fmax)
    fi = freqs[fm]
    if len(fi) == 0: return []
    w = np.hanning(ws).astype(np.float32)
    bs = float(fi[1] - fi[0]) if len(fi) > 1 else 1.0
    out = []
    for pos in positions:
        sp = np.abs(np.fft.rfft(y[pos:pos+ws] * w))[fm]
        if np.max(sp) < 1e-4: continue
        pi = int(np.argmax(sp))
        pf = float(fi[pi])
        if 0 < pi < len(sp)-1:
            y0, y1, y2 = float(sp[pi-1]), float(sp[pi]), float(sp[pi+1])
            d = y0 - 2*y1 + y2
            if abs(d) > 1e-9:
                pf += 0.5*(y0-y2)/d * bs
        out.append(pf)
    return out


def profil_frasa(y, sr, cfg):
    ps = detect_pitch_fft(y, sr, cfg['fmin'], cfg['fmax'])
    if len(ps) < 3: return None
    nl = cfg['nada']; nm = cfg['nama_nada']
    def k(f):
        if np.isnan(f): return -1
        return int(np.argmin([abs(f - n) for n in nl]))
    cs = [k(p) for p in ps]
    v = [c for c in cs if c >= 0]
    if not v: return None
    dist = {nm[i]: 100.0*sum(1 for c in v if c==i)/len(v) for i in range(len(nm))}
    return {'dist': dist, 'nada_awal': nm[k(ps[0])],
            'nada_akhir': nm[k(ps[-1])], 'durasi_ms': len(y)/sr*1000.0}


print("\n📁 Loading metadata...")
t0 = time.time()
POOLS_META = {}
GLOBAL_SR = 44100

for kode, cfg in INSTRUMEN.items():
    print(f"\n  📂 {cfg['nama']}...")
    pool = {}
    folder = os.path.join(DATA_DIR, cfg['folder'])
    if not os.path.exists(folder):
        print(f"     ⚠️ Tidak ada folder")
        POOLS_META[kode] = {}
        continue

    if cfg['mode'] == 'raw_files':
        for kategori, fname in cfg['files'].items():
            path = os.path.join(folder, fname)
            if not os.path.exists(path): continue
            y_t, sr = sf.read(path, dtype='float32')
            GLOBAL_SR = sr
            if len(y_t.shape) > 1: y_t = y_t.mean(axis=1)
            ph = detect_phrases_from_file(y_t, sr)
            pool[kategori] = []
            for s, e in ph:
                prof = profil_frasa(y_t[s:e], sr, cfg)
                if prof:
                    pool[kategori].append({
                        'path': path, 'start': int(s), 'end': int(e),
                        'sr': sr, 'profil': prof,
                        'nama': f"{kategori}_{len(pool[kategori]):02d}",
                    })
            del y_t; gc.collect()
            print(f"     {kategori:12s}: {len(pool[kategori])} frasa")
    else:
        pj = os.path.join(DATA_DIR, cfg['pool_json'])
        if not os.path.exists(pj):
            print(f"     ⚠️ Pool JSON tidak ada")
            POOLS_META[kode] = {}
            continue
        with open(pj) as f: pd = json.load(f)
        all_f = []
        for it in pd:
            p = os.path.join(folder, it['file'])
            if not os.path.exists(p): continue
            all_f.append({
                'path': p, 'sr': 44100,
                'profil': {'dist': it['dist'], 'nada_awal': it['nada_awal'],
                           'nada_akhir': it['nada_akhir'], 'durasi_ms': it['durasi']*1000},
                'nama': it['file'].replace('.wav', ''),
            })
        pb, pt = [], []
        for fr in all_f:
            if cfg['kode'] == 'ampek':
                if fr['profil']['nada_awal'] in ['P1', 'P2']: pb.append(fr)
                if fr['profil']['nada_akhir'] in ['A1', 'A2']: pt.append(fr)
            elif cfg['kode'] == 'bansi':
                if fr['profil']['nada_awal'] in ['B7','B8','B9','B10','B11']: pb.append(fr)
                if fr['profil']['nada_akhir'] in ['B1','B2']: pt.append(fr)
        pool['PEMBUKAAN'] = pb if pb else all_f[:50]
        pool['PENUTUPAN'] = pt if pt else all_f[-50:]
        pool['ISI'] = all_f
        pool['GURAU'] = all_f
        pool['SEDIH'] = all_f
        pool['DUODUO'] = all_f
        print(f"     PEMBUKAAN   : {len(pool['PEMBUKAAN'])} frasa")
        print(f"     ISI         : {len(pool['ISI'])} frasa")
        print(f"     PENUTUPAN   : {len(pool['PENUTUPAN'])} frasa")
    POOLS_META[kode] = pool

total = sum(len(p) for pool in POOLS_META.values() for p in pool.values())
print(f"\n✅ {total} frasa metadata dimuat dalam {time.time()-t0:.1f}s")
gc.collect()


app = FastAPI(title="Nalasaluang API v3.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"status": "ok", "service": "Nalasaluang API", "version": "3.1"}


@app.get("/api/health")
async def health():
    r = {'status': 'ok', 'instrumen': {}, 'cache': len(_frasa_cache)}
    for k, p in POOLS_META.items():
        r['instrumen'][k] = {kk: len(v) for kk, v in p.items()}
    return r


@app.get("/audio/{filename}")
async def audio(filename: str):
    p = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(p): return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="audio/wav")


def skor_genre(dist, genre, kode):
    if kode == 'ampek':
        if genre == 'sedih': return dist.get('A1',0)*1.5 + dist.get('A2',0)*1.2
        elif genre == 'gurau': return dist.get('A3',0) + dist.get('A4',0)*1.5 + dist.get('P1',0) + dist.get('P2',0)
        else: return dist.get('A3',0) + dist.get('A1',0) + dist.get('A2',0)
    elif kode == 'bansi':
        if genre == 'sedih': return dist.get('B1',0)*2 + dist.get('B2',0)*1.5 + dist.get('B3',0)
        elif genre == 'gurau': return dist.get('B7',0) + dist.get('B8',0) + dist.get('B9',0) + dist.get('B10',0)*1.5 + dist.get('B11',0)*1.5
        else: return dist.get('B3',0) + dist.get('B4',0) + dist.get('B5',0) + dist.get('B6',0)
    else:
        if genre == 'sedih': return dist.get('N1',0)*1.5 + (dist.get('N5',0)+dist.get('N6',0))*1.2 - (dist.get('N3',0)+dist.get('N4',0))*2.0
        elif genre == 'gurau': return dist.get('N8',0)*1.3 + dist.get('N3',0)*1.2 + dist.get('N7',0)*1.2 - dist.get('N6',0)*2.0
        else: return dist.get('N7',0) + dist.get('N1',0) + dist.get('N8',0) + dist.get('N2',0)


def pilih_pembukaan(pool, pola, kode, n=3):
    fr = pool.get('PEMBUKAAN', []) or pool.get('ISI', [])
    if not fr: return []
    fr = fr.copy()
    if kode == 'bansi':
        if pola == 'Megah': fr.sort(key=lambda x: -sum(x['profil']['dist'].get(k,0) for k in ['B9','B10','B11']))
        elif pola == 'Tenang': fr.sort(key=lambda x: -sum(x['profil']['dist'].get(k,0) for k in ['B1','B2','B3']))
        elif pola == 'Tegang': fr.sort(key=lambda x: -sum(x['profil']['dist'].get(k,0) for k in ['B10','B11']))
    elif kode == 'ampek':
        if pola == 'Megah': fr.sort(key=lambda x: -(x['profil']['dist'].get('P1',0)+x['profil']['dist'].get('P2',0)))
        elif pola == 'Tenang': fr.sort(key=lambda x: -(x['profil']['dist'].get('A1',0)+x['profil']['dist'].get('A2',0)))
        elif pola == 'Tegang': fr.sort(key=lambda x: -(x['profil']['dist'].get('P2',0)*1.5+x['profil']['dist'].get('P1',0)))
    else:
        if pola == 'Megah': fr.sort(key=lambda x: -(x['profil']['dist'].get('N6',0)+x['profil']['dist'].get('N5',0)))
        elif pola == 'Tenang': fr.sort(key=lambda x: -(x['profil']['dist'].get('N1',0)+x['profil']['dist'].get('N2',0)))
        elif pola == 'Tegang': fr.sort(key=lambda x: -(x['profil']['dist'].get('N7',0)+x['profil']['dist'].get('N8',0)))
    return fr[:n]


def pilih_penutupan(pool, pola, kode, n=3):
    fr = pool.get('PENUTUPAN', []) or pool.get('ISI', [])
    if not fr: return []
    fr = fr.copy()
    if kode == 'bansi':
        if pola == 'Tenang': fr.sort(key=lambda x: -sum(x['profil']['dist'].get(k,0) for k in ['B1','B2']))
        elif pola == 'Tegas': fr.sort(key=lambda x: -x['profil']['dist'].get('B1',0))
        elif pola == 'Menggantung': fr.sort(key=lambda x: -sum(x['profil']['dist'].get(k,0) for k in ['B3','B4']))
    elif kode == 'ampek':
        if pola == 'Tenang': fr.sort(key=lambda x: -(x['profil']['dist'].get('A1',0)+x['profil']['dist'].get('A2',0)))
        elif pola == 'Tegas': fr.sort(key=lambda x: -x['profil']['dist'].get('A1',0))
        elif pola == 'Menggantung': fr.sort(key=lambda x: -(x['profil']['dist'].get('A3',0)+x['profil']['dist'].get('A2',0)))
    else:
        fr = [f for f in fr if f['profil']['nada_akhir'] == 'N1'] or fr
        if pola == 'Tenang': fr.sort(key=lambda x: -(x['profil']['dist'].get('N1',0)+x['profil']['dist'].get('N7',0)))
        elif pola == 'Tegas': fr.sort(key=lambda x: -x['profil']['dist'].get('N1',0))
        elif pola == 'Menggantung': fr.sort(key=lambda x: -(x['profil']['dist'].get('N5',0)+x['profil']['dist'].get('N6',0)))
    return fr[:n]


def susun_isi(pool, genre, target_s, kode, cf=0.08):
    key = {'sedih':'SEDIH', 'gurau':'GURAU', 'biasa':'DUODUO'}[genre]
    fr = pool.get(key, []) or pool.get('ISI', [])
    if not fr: return []
    fr = fr.copy()
    for p in fr: p['skor'] = skor_genre(p['profil']['dist'], genre, kode)
    fr.sort(key=lambda x: -x['skor'])
    fr = fr[:max(3, int(len(fr) * 0.6))]
    random.shuffle(fr)
    hasil = [fr[0]]; sisa = fr[1:]
    dur = hasil[0]['profil']['durasi_ms']/1000
    loops, mx = 0, 1000
    while dur < target_s and loops < mx:
        if not sisa:
            loops += 1; sisa = list(fr); random.shuffle(sisa)
        ak = hasil[-1]['profil']['nada_akhir']
        nm = list(hasil[-1]['profil']['dist'].keys())
        ia = nm.index(ak) if ak in nm else 0
        sisa.sort(key=lambda x: abs(nm.index(x['profil']['nada_awal']) - ia) if x['profil']['nada_awal'] in nm else 99)
        pl = sisa[0]; hasil.append(pl); sisa.pop(0)
        dur += pl['profil']['durasi_ms']/1000 - cf
    return hasil


def get_segmen(fi):
    if 'path' not in fi: return None
    lo = load_frasa(fi['path'])
    if lo is None: return None
    yi, sr = lo
    if 'start' in fi:
        return int16_to_float(yi[fi['start']:fi['end']])
    return int16_to_float(yi)


def render(fl, cf_ms=80):
    if not fl: return np.array([], dtype=np.float32)
    CF = int(cf_ms * GLOBAL_SR / 1000)
    segs = []
    for f in fl:
        s = get_segmen(f)
        if s is not None and len(s) > 0: segs.append(s)
    if not segs: return np.array([], dtype=np.float32)
    if len(segs) == 1: return segs[0]
    total = sum(len(s) for s in segs) - CF * (len(segs) - 1)
    res = np.zeros(total, dtype=np.float32)
    fi = np.linspace(0, 1, CF, dtype=np.float32) ** 0.5
    fo = np.linspace(1, 0, CF, dtype=np.float32) ** 0.5
    pos = 0
    for i, seg in enumerate(segs):
        seg = seg.copy()
        if i < len(segs)-1 and len(seg) > CF: seg[-CF:] *= fo
        if i > 0 and len(seg) > CF: seg[:CF] *= fi
        if i == 0:
            res[:len(seg)] += seg; pos = len(seg) - CF
        else:
            ep = pos + len(seg)
            if ep > len(res):
                seg = seg[:len(res)-pos]; ep = len(res)
            res[pos:ep] += seg; pos = ep - CF
    pk = float(np.max(np.abs(res)))
    if pk > 0: res = res / pk * 0.9
    return res


class GenerateRequest(BaseModel):
    instrumen: str
    pola_bukak: str
    genre: str
    pola_tutup: str
    durasi_menit: int


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        if req.instrumen not in POOLS_META:
            return JSONResponse({"error": "Instrumen tidak dikenal"}, status_code=400)
        pool = POOLS_META[req.instrumen]
        if not pool:
            return JSONResponse({"error": "Pool kosong"}, status_code=500)
        ts = req.durasi_menit * 60
        pb = pilih_pembukaan(pool, req.pola_bukak, req.instrumen, 3)
        pt = pilih_penutupan(pool, req.pola_tutup, req.instrumen, 3)
        dpb = sum(p['profil']['durasi_ms'] for p in pb)/1000 - 0.08*max(0, len(pb)-1)
        dpt = sum(p['profil']['durasi_ms'] for p in pt)/1000 - 0.08*max(0, len(pt)-1)
        ti = max(10, ts - dpb - dpt)
        isi = susun_isi(pool, req.genre, ti, req.instrumen)
        if not (pb and isi and pt):
            return JSONResponse({"error": "Tidak cukup frasa"}, status_code=500)
        final = render(pb + isi + pt)
        gc.collect()
        fn = f"nalasaluang_{req.instrumen}_{int(time.time())}.wav"
        op = os.path.join(OUTPUT_DIR, fn)
        sf.write(op, final, GLOBAL_SR, subtype='PCM_16')
        return {"audio_url": f"/audio/{fn}", "durasi": round(len(final)/GLOBAL_SR, 1),
                "frasa_pb": len(pb), "frasa_isi": len(isi), "frasa_pt": len(pt),
                "instrumen": req.instrumen}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
