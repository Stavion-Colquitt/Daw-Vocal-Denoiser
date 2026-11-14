"""Held-out test through the STREAMING engine (exactly what a DAW would run).
Test vocals: the artist's held-out song (Say Dat Name takes) + VocalSet test singers (female9, male11), mixed with
dev-split DNS noise / bleed / hiss / hum at SNRs 0..40 dB from a fixed seed. Reports, noisy vs enhanced:
  SI-SDR (dB), PESQ-WB (at 16 kHz), and "clean pass-through" SI-SDR: clean vocals in -> how untouched they come out.
    py -3.12 eval.py [n_items]
"""
import sys, os, glob, random, json, numpy as np
from scipy.signal import resample_poly
from pesq import pesq
import dataset as D
from stream import process_file

def si_sdr(r, e):
    r, e = r - r.mean(), e - e.mean(); t = np.dot(e, r) / (np.dot(r, r) + 1e-12) * r
    return 10 * np.log10((t ** 2).sum() / (((e - t) ** 2).sum() + 1e-12) + 1e-12)

def p16(c, y):
    try: return pesq(16000, resample_poly(c, 1, 3), resample_poly(y, 1, 3), "wb")
    except Exception: return np.nan

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    D.CLEAN_W = {"own": 0.5, "vocalset": 0.5, "speech": 0.0}
    ds = D.Mixtures("dev", n, seed=777)
    ds.clean = {"own": sorted(glob.glob(os.path.join(D.DATA, "own_vocals", "test", "*.wav"))),
                "vocalset": sorted(glob.glob(os.path.join(D.DATA, "vocalset", "test", "*.wav"))), "speech": []}
    ds.kinds = [k for k in ("own", "vocalset") if ds.clean[k]]; ds.w = np.ones(len(ds.kinds)) / len(ds.kinds)
    R = {k: [] for k in ("sisdr_n", "sisdr_e", "pesq_n", "pesq_e", "passthru")}
    for i in range(n):
        x, c = (t.numpy() for t in ds[i]); y = process_file(x)
        if np.allclose(x, c):
            R["passthru"].append(si_sdr(c, y)); continue
        R["sisdr_n"].append(si_sdr(c, x)); R["sisdr_e"].append(si_sdr(c, y))
        R["pesq_n"].append(p16(c, x)); R["pesq_e"].append(p16(c, y))
    m = {k: float(np.nanmean(v)) if v else None for k, v in R.items()}
    print(f"held-out singing ({n} items: {', '.join(ds.kinds)})")
    print(f"SI-SDR   noisy {m['sisdr_n']:.2f} dB -> enhanced {m['sisdr_e']:.2f} dB  ({m['sisdr_e'] - m['sisdr_n']:+.2f})")
    print(f"PESQ-WB  noisy {m['pesq_n']:.3f} -> enhanced {m['pesq_e']:.3f}  ({m['pesq_e'] - m['pesq_n']:+.3f})")
    if m["passthru"] is not None: print(f"clean pass-through SI-SDR {m['passthru']:.1f} dB  (higher = clean vocals left alone)")
    json.dump(m, open("eval_results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
