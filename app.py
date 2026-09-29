"""
NALASALUANG Backend v3.0 — Cloud Edition
Mendukung Patiak Tigo & Patiak Ampek & Bansi.
Download frasa otomatis dari Hugging Face Dataset saat startup.
"""
import os
import random
import time
import json
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import snapshot_download

# ============================================================
# KONFIGURASI
# ============================================================
HF_DATASET = "Nalareka/nalasaluang-frasa"
DATA_DIR = "/tmp/nalasaluang_data"
OUTPUT_DIR = "/tmp/nalasaluang_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INSTRUMEN = {
    'tigo': {
        'kode': 'tigo', 'nama': 'Patiak Tigo', 'subtitle': 'Solok Selatan',
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
        'kode': 'ampek', 'nama': 'Patiak Ampek', 'subtitle': 'Darek',
        'nada': [229.0, 241.9, 262.1, 309.9, 460.5, 629.6],
        'nama_nada': ['A1', 'A2', 'A3', 'A4', 'P1', 'P2'],
        'mode': 'frasa', 'folder': 'patiak_ampek_frasa',
        'files': None, 'pool_json': 'pakem_patiak_ampek_pool.json',
        'fmin': 180, 'fmax': 700,
    },
    'bansi': {
        'kode': 'bansi', 'nama': 'Bansi', 'subtitle': 'Talang Diatonik',
        'nada': [479.0, 534.7, 598.3, 630.1, 658.0, 701.7, 793.2, 844.9, 888.7, 952.3, 1067.7],
        'nama_nada': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11'],
        'mode': 'frasa', 'folder': 'bansi_frasa',
        'files': None, 'pool_json': 'bansi_pool.json',
        'fmin': 400, 'fmax': 1200,
    },
}

# ============================================================
# DOWNLOAD DATASET DARI HF
# ============================================================
print("=" * 60)
print("NALASALUANG Backend v3.0 — Cloud Edition")
print("=" * 60)
print(f"\n📥 Download dataset dari Hugging Face: {HF_DATASET}")
t0 = time.time()
try:
    snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"✅ Dataset downloaded dalam {time.time()-t0:.1f}s")
except Exception as e:
    print(f"⚠️  Download gagal: {e}")
    print("   Backend akan coba jalan dengan dataset yang ada.")

# ============================================================
# UTILITIES
# ============================================================
def compute_rms(y, frame_length=1024, hop_length=512):
    n_frames = (len(y) - frame_length) // hop_length + 1
    if n_frames <= 0: return np.array([], dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        pos = i * hop_length
        rms[i] = np.sqrt(np.mean(y[pos:pos+frame_length]**2))
    return rms

def detect_phrases_from_file(y, sr, top_db=35, min_silence=0.3, min_phrase=1.5):
    rms = compute_rms(y)
    if len(rms) == 0: return []
    max_rms = float(np.max(rms))
    if max_rms < 1e-6: return []
    thr = max_rms * (10 ** (-top_db / 20.0))
    is_sp = rms > thr
    hop = 512
    phrases, in_p, start_t = [], False, 0.0
    for i, s in enumerate(is_sp):
        t = i * hop / sr
        if s and not in_p: start_t = t; in_p = True
        elif not s and in_p:
            if t - start_t >= min_phrase: phrases.append((int(start_t*sr), int(t*sr)))
            in_p = False
    if in_p and len(y)/sr - start_t >= min_phrase:
        phrases.append((int(start_t*sr), len(y)))
    merged = []
    for s, e in phrases:
        if merged and (s - merged[-1][1])/sr < min_silence:
            merged[-1] = (merged[-1][0], e)
        else: merged.append((s, e))
    return merged

def detect_pitch_fft(y, sr, fmin, fmax):
    window_size = 4096
    if len(y) < window_size: return []
    positions = np.linspace(0, len(y) - window_size, 
                            min(20, max(1, len(y)//window_size)), dtype=int)
    freqs = np.fft.rfftfreq(window_size, 1.0/sr)
    fm = (freqs >= fmin) & (freqs <= fmax)
    freqs_in = freqs[fm]
    if len(freqs_in) == 0: return []
    window = np.hanning(window_size).astype(np.float32)
    bin_sz = float(freqs_in[1] - freqs_in[0]) if len(freqs_in) > 1 else 1.0
    pitches = []
    for pos in positions:
        frame = y[pos:pos+window_size] * window
        sp = np.abs(np.fft.rfft(frame))[fm]
        if np.max(sp) < 1e-4: continue
        pi = int(np.argmax(sp))
        pf = float(freqs_in[pi])
        if 0 < pi < len(sp)-1:
            y0, y1, y2 = float(sp[pi-1]), float(sp[pi]), float(sp[pi+1])
            d = y0 - 2*y1 + y2
            if abs(d) > 1e-9:
                pf += 0.5*(y0-y2)/d * bin_sz
        pitches.append(pf)
    return pitches

def profil_frasa(y, sr, cfg):
    pitches = detect_pitch_fft(y, sr, cfg['fmin'], cfg['fmax'])
    if len(pitches) < 3: return None
    nada_list = cfg['nada']
    nama_list = cfg['nama_nada']
    def klas(f):
        if np.isnan(f): return -1
        return int(np.argmin([abs(f - n) for n in nada_list]))
    classes = [klas(p) for p in pitches]
    valid = [c for c in classes if c >= 0]
    if not valid: return None
    dist = {nama_list[i]: 100.0*sum(1 for c in valid if c==i)/len(valid) 
            for i in range(len(nama_list))}
    return {'dist': dist, 'nada_awal': nama_list[klas(pitches[0])],
            'nada_akhir': nama_list[klas(pitches[-1])], 'durasi_ms': len(y)/sr*1000.0}

# ============================================================
# LOAD POOLS
# ============================================================
print("\n📁 Loading instrumen...")
t0 = time.time()
POOLS = {}
GLOBAL_SR = 44100

for kode, cfg in INSTRUMEN.items():
    print(f"\n  📂 {cfg['nama']}...")
    pool = {}
    folder = os.path.join(DATA_DIR, cfg['folder'])
    
    if not os.path.exists(folder):
        print(f"     ⚠️ Folder tidak ada: {folder}")
        POOLS[kode] = {}
        continue
    
    if cfg['mode'] == 'raw_files':
        for kategori, fname in cfg['files'].items():
            path = os.path.join(folder, fname)
            if not os.path.exists(path):
                print(f"     ⚠️ Skip {fname}")
                continue
            y, sr = sf.read(path)
            GLOBAL_SR = sr
            if len(y.shape) > 1: y = y.mean(axis=1)
            y = y.astype(np.float32)
            phrases = detect_phrases_from_file(y, sr)
            pool[kategori] = []
            for s, e in phrases:
                seg = y[s:e].copy()
                profil = profil_frasa(seg, sr, cfg)
                if profil:
                    pool[kategori].append({
                        'y': seg, 'sr': sr, 'profil': profil,
                        'nama': f"{kategori}_{len(pool[kategori]):02d}",
                    })
            print(f"     {kategori:12s}: {len(pool[kategori])} frasa")
    
    elif cfg['mode'] == 'frasa':
        pool_json_path = os.path.join(DATA_DIR, cfg['pool_json'])
        if not os.path.exists(pool_json_path):
            print(f"     ⚠️ Pool JSON tidak ada: {pool_json_path}")
            POOLS[kode] = {}
            continue
        
        with open(pool_json_path) as f:
            pool_data = json.load(f)
        
        all_frasa = []
        for item in pool_data:
            path = os.path.join(folder, item['file'])
            if not os.path.exists(path): continue
            y, sr = sf.read(path)
            if len(y.shape) > 1: y = y.mean(axis=1)
            y = y.astype(np.float32)
            all_frasa.append({
                'y': y, 'sr': sr,
                'profil': {
                    'dist': item['dist'],
                    'nada_awal': item['nada_awal'],
                    'nada_akhir': item['nada_akhir'],
                    'durasi_ms': item['durasi'] * 1000,
                },
                'nama': item['file'].replace('.wav', ''),
            })
        
        pb_list, pt_list = [], []
        for frasa in all_frasa:
            if cfg['kode'] == 'ampek':
                if frasa['profil']['nada_awal'] in ['P1', 'P2']:
                    pb_list.append(frasa)
                if frasa['profil']['nada_akhir'] in ['A1', 'A2']:
                    pt_list.append(frasa)
            elif cfg['kode'] == 'bansi':
                if frasa['profil']['nada_awal'] in ['B7', 'B8', 'B9', 'B10', 'B11']:
                    pb_list.append(frasa)
                if frasa['profil']['nada_akhir'] in ['B1', 'B2']:
                    pt_list.append(frasa)
        
        pool['PEMBUKAAN'] = pb_list if pb_list else all_frasa[:50]
        pool['PENUTUPAN'] = pt_list if pt_list else all_frasa[-50:]
        pool['ISI'] = all_frasa
        pool['GURAU'] = all_frasa
        pool['SEDIH'] = all_frasa
        pool['DUODUO'] = all_frasa
        
        print(f"     PEMBUKAAN   : {len(pool['PEMBUKAAN'])} frasa")
        print(f"     ISI         : {len(pool['ISI'])} frasa")
        print(f"     PENUTUPAN   : {len(pool['PENUTUPAN'])} frasa")
    
    POOLS[kode] = pool

total = sum(len(p) for pool in POOLS.values() for p in pool.values())
print(f"\n✅ Total {total} frasa dimuat dalam {time.time()-t0:.1f}s")

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI(title="Nalasaluang API v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "service": "Nalasaluang API", "version": "3.0"}

@app.get("/api/health")
async def health():
    result = {'status': 'ok', 'instrumen': {}}
    for kode, pool in POOLS.items():
        result['instrumen'][kode] = {k: len(v) for k, v in pool.items()}
    return result

@app.get("/audio/{filename}")
async def audio(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")

# ============================================================
# GENERATE LOGIC
# ============================================================
def skor_genre(dist, genre, kode):
    if kode == 'ampek':
        if genre == 'sedih':
            return dist.get('A1', 0)*1.5 + dist.get('A2', 0)*1.2
        elif genre == 'gurau':
            return (dist.get('A3', 0) + dist.get('A4', 0)*1.5 
                    + dist.get('P1', 0) + dist.get('P2', 0))
        else:
            return dist.get('A3', 0) + dist.get('A1', 0) + dist.get('A2', 0)
    elif kode == 'bansi':
        if genre == 'sedih':
            return dist.get('B1', 0)*2 + dist.get('B2', 0)*1.5 + dist.get('B3', 0)
        elif genre == 'gurau':
            return (dist.get('B7', 0) + dist.get('B8', 0) + dist.get('B9', 0) 
                    + dist.get('B10', 0)*1.5 + dist.get('B11', 0)*1.5)
        else:
            return dist.get('B3', 0) + dist.get('B4', 0) + dist.get('B5', 0) + dist.get('B6', 0)
    else:
        if genre == 'sedih':
            return (dist.get('N1', 0)*1.5 + (dist.get('N5', 0)+dist.get('N6', 0))*1.2
                    - (dist.get('N3', 0)+dist.get('N4', 0))*2.0)
        elif genre == 'gurau':
            return (dist.get('N8', 0)*1.3 + dist.get('N3', 0)*1.2 
                    + dist.get('N7', 0)*1.2 - dist.get('N6', 0)*2.0)
        else:
            return dist.get('N7', 0) + dist.get('N1', 0) + dist.get('N8', 0) + dist.get('N2', 0)

def pilih_pembukaan(pool, pola, kode, n=3):
    frasa = pool.get('PEMBUKAAN', []) or pool.get('ISI', [])
    if not frasa: return []
    frasa = frasa.copy()
    if kode == 'bansi':
        if pola == 'Megah':
            frasa.sort(key=lambda x: -sum(x['profil']['dist'].get(k, 0) for k in ['B9','B10','B11']))
        elif pola == 'Tenang':
            frasa.sort(key=lambda x: -sum(x['profil']['dist'].get(k, 0) for k in ['B1','B2','B3']))
        elif pola == 'Tegang':
            frasa.sort(key=lambda x: -sum(x['profil']['dist'].get(k, 0) for k in ['B10','B11']))
    elif kode == 'ampek':
        if pola == 'Megah':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('P1', 0) + x['profil']['dist'].get('P2', 0)))
        elif pola == 'Tenang':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('A1', 0) + x['profil']['dist'].get('A2', 0)))
        elif pola == 'Tegang':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('P2', 0)*1.5 + x['profil']['dist'].get('P1', 0)))
    else:
        if pola == 'Megah':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('N6', 0) + x['profil']['dist'].get('N5', 0)))
        elif pola == 'Tenang':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('N1', 0) + x['profil']['dist'].get('N2', 0)))
        elif pola == 'Tegang':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('N7', 0) + x['profil']['dist'].get('N8', 0)))
    return frasa[:n]

def pilih_penutupan(pool, pola, kode, n=3):
    frasa = pool.get('PENUTUPAN', []) or pool.get('ISI', [])
    if not frasa: return []
    frasa = frasa.copy()
    if kode == 'bansi':
        if pola == 'Tenang':
            frasa.sort(key=lambda x: -sum(x['profil']['dist'].get(k, 0) for k in ['B1','B2']))
        elif pola == 'Tegas':
            frasa.sort(key=lambda x: -x['profil']['dist'].get('B1', 0))
        elif pola == 'Menggantung':
            frasa.sort(key=lambda x: -sum(x['profil']['dist'].get(k, 0) for k in ['B3','B4']))
    elif kode == 'ampek':
        if pola == 'Tenang':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('A1', 0) + x['profil']['dist'].get('A2', 0)))
        elif pola == 'Tegas':
            frasa.sort(key=lambda x: -x['profil']['dist'].get('A1', 0))
        elif pola == 'Menggantung':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('A3', 0) + x['profil']['dist'].get('A2', 0)))
    else:
        frasa = [f for f in frasa if f['profil']['nada_akhir'] == 'N1'] or frasa
        if pola == 'Tenang':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('N1', 0) + x['profil']['dist'].get('N7', 0)))
        elif pola == 'Tegas':
            frasa.sort(key=lambda x: -x['profil']['dist'].get('N1', 0))
        elif pola == 'Menggantung':
            frasa.sort(key=lambda x: -(x['profil']['dist'].get('N5', 0) + x['profil']['dist'].get('N6', 0)))
    return frasa[:n]

def susun_isi(pool, genre, target_s, kode, cf=0.08):
    key = {'sedih':'SEDIH', 'gurau':'GURAU', 'biasa':'DUODUO'}[genre]
    frasa = pool.get(key, []) or pool.get('ISI', [])
    if not frasa: return []
    frasa = frasa.copy()
    for p in frasa:
        p['skor'] = skor_genre(p['profil']['dist'], genre, kode)
    frasa.sort(key=lambda x: -x['skor'])
    frasa = frasa[:max(3, int(len(frasa) * 0.6))]
    random.shuffle(frasa)
    hasil = [frasa[0]]
    sisa = frasa[1:]
    dur = hasil[0]['profil']['durasi_ms']/1000
    max_loops, loops = 1000, 0
    while dur < target_s and loops < max_loops:
        if not sisa:
            loops += 1
            sisa = list(frasa)
            random.shuffle(sisa)
        akhir = hasil[-1]['profil']['nada_akhir']
        nama_list = list(hasil[-1]['profil']['dist'].keys())
        ia = nama_list.index(akhir) if akhir in nama_list else 0
        sisa.sort(key=lambda x: abs(nama_list.index(x['profil']['nada_awal']) - ia) 
                  if x['profil']['nada_awal'] in nama_list else 99)
        pilih = sisa[0]
        hasil.append(pilih)
        sisa.pop(0)
        dur += pilih['profil']['durasi_ms']/1000 - cf
    return hasil

def render(frasa_list, cf_ms=80):
    if not frasa_list: return np.array([], dtype=np.float32)
    if len(frasa_list) == 1: return frasa_list[0]['y']
    CF = int(cf_ms * GLOBAL_SR / 1000)
    segs = [f['y'] for f in frasa_list]
    total = sum(len(s) for s in segs) - CF * (len(segs) - 1)
    result = np.zeros(total, dtype=np.float32)
    fi = np.linspace(0, 1, CF, dtype=np.float32) ** 0.5
    fo = np.linspace(1, 0, CF, dtype=np.float32) ** 0.5
    pos = 0
    for i, seg in enumerate(segs):
        seg = seg.copy()
        if i < len(segs)-1 and len(seg) > CF: seg[-CF:] *= fo
        if i > 0 and len(seg) > CF: seg[:CF] *= fi
        if i == 0:
            result[:len(seg)] += seg
            pos = len(seg) - CF
        else:
            ep = pos + len(seg)
            if ep > len(result):
                seg = seg[:len(result)-pos]; ep = len(result)
            result[pos:ep] += seg
            pos = ep - CF
    peak = float(np.max(np.abs(result)))
    if peak > 0: result = result / peak * 0.9
    return result

class GenerateRequest(BaseModel):
    instrumen: str
    pola_bukak: str
    genre: str
    pola_tutup: str
    durasi_menit: int

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    try:
        if req.instrumen not in POOLS:
            return JSONResponse({"error": f"Instrumen {req.instrumen} tidak dikenal"}, status_code=400)
        pool = POOLS[req.instrumen]
        if not pool:
            return JSONResponse({"error": f"Pool {req.instrumen} kosong"}, status_code=500)
        target_s = req.durasi_menit * 60
        pb = pilih_pembukaan(pool, req.pola_bukak, req.instrumen, 3)
        pt = pilih_penutupan(pool, req.pola_tutup, req.instrumen, 3)
        dur_pb = sum(p['profil']['durasi_ms'] for p in pb)/1000 - 0.08*max(0, len(pb)-1)
        dur_pt = sum(p['profil']['durasi_ms'] for p in pt)/1000 - 0.08*max(0, len(pt)-1)
        target_isi = max(10, target_s - dur_pb - dur_pt)
        isi = susun_isi(pool, req.genre, target_isi, req.instrumen)
        if not (pb and isi and pt):
            return JSONResponse({"error": "Tidak cukup frasa"}, status_code=500)
        final = render(pb + isi + pt)
        ts = int(time.time())
        fname = f"nalasaluang_{req.instrumen}_{ts}.wav"
        out_path = os.path.join(OUTPUT_DIR, fname)
        sf.write(out_path, final, GLOBAL_SR, subtype='PCM_16')
        return {
            "audio_url": f"/audio/{fname}",
            "durasi": round(len(final)/GLOBAL_SR, 1),
            "frasa_pb": len(pb),
            "frasa_isi": len(isi),
            "frasa_pt": len(pt),
            "instrumen": req.instrumen,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
