"""Held-out test through the STREAMING engine (exactly what a DAW would run).
Test vocals: the artist's held-out song (Say Dat Name takes) + VocalSet test singers (female9, male11), mixed with
dev-split DNS noise / bleed / hiss / hum at SNRs 0..40 dB from a fixed seed. Reports, noisy vs enhanced:

  Speech-style metrics (kept for comparison with earlier versions):
    SI-SDR (dB), PESQ-WB (at 16 kHz; designed for speech, so only a rough guide on singing),
    "clean pass-through" SI-SDR: clean vocals in -> how untouched they come out.
  Music / singing metrics (what the singing-separation literature recommends):
    SDR (dB)        filter-based BSS-Eval SDR (512-tap distortion filter), more tolerant of small EQ/phase changes
    ViSQOL (1-5)    Google ViSQOL v3, audio mode, 48 kHz full band (runs in Docker: tools/visqol)
    MERT-MSE        distance between MERT-v1-95M music embeddings of output and clean reference (lower = closer)
    FAD-MERT        Frechet distance between the MERT frame-embedding distributions of all outputs vs all clean refs
  Signals are loudness-normalized (EBU R128, -23 LUFS) before ViSQOL and MERT so level differences don't bias them.
  Breakdowns (mean improvement, 95% CI) by input SNR band, noise type and vocal source; per-item rows go to a CSV.

    py -3.12 eval.py [n_items] [--no-visqol] [--no-mert] [--out eval_results.json]
"""
import sys, os, glob, json, csv, argparse, tempfile, subprocess, numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pesq import pesq
import dataset as D
from stream import process_file

SR = 48000
SNR_BANDS = [(0, 10), (10, 20), (20, 30), (30, 40.01)]
MERT_ID = "m-a-p/MERT-v1-95M"


def si_sdr(r, e):
    r, e = r - r.mean(), e - e.mean(); t = np.dot(e, r) / (np.dot(r, r) + 1e-12) * r
    return 10 * np.log10((t ** 2).sum() / (((e - t) ** 2).sum() + 1e-12) + 1e-12)

def p16(c, y):
    try: return pesq(16000, resample_poly(c, 1, 3), resample_poly(y, 1, 3), "wb")
    except Exception: return np.nan

def sdr(r, e):
    import fast_bss_eval
    try: return float(fast_bss_eval.sdr(r[None].astype(np.float64), e[None].astype(np.float64), filter_length=512)[0])
    except Exception: return np.nan

_meter = None
def lufs_norm(x, target=-23.0):
    """EBU R128 integrated loudness normalization; peak-limited to 0.99 so 16-bit WAVs for ViSQOL don't clip."""
    global _meter
    import pyloudnorm
    _meter = _meter or pyloudnorm.Meter(SR)
    l = _meter.integrated_loudness(x.astype(np.float64))
    y = x * 10 ** ((target - l) / 20) if np.isfinite(l) else x
    pk = np.abs(y).max()
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


# ---------------- ViSQOL (Docker) ----------------
def visqol_batch(pairs, image="visqol:local"):
    """pairs: list of (ref, deg) float32 arrays at 48 kHz. Returns MOS-LQO list (1-5), audio mode."""
    here = os.path.dirname(os.path.abspath(__file__))
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        print(f"building {image} from tools/visqol (one time, several minutes)...", flush=True)
        subprocess.run(["docker", "build", "-t", image, os.path.join(here, "tools", "visqol")], check=True)
    with tempfile.TemporaryDirectory() as td:
        rows = []
        for i, (r, d) in enumerate(pairs):
            sf.write(os.path.join(td, f"r{i}.wav"), r, SR, subtype="PCM_16")
            sf.write(os.path.join(td, f"d{i}.wav"), d, SR, subtype="PCM_16")
            rows.append((f"/work/r{i}.wav", f"/work/d{i}.wav"))
        with open(os.path.join(td, "pairs.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["reference", "degraded"]); w.writerows(rows)
        subprocess.run(["docker", "run", "--rm", "-v", f"{td}:/work", image, "--batch_input_csv", "/work/pairs.csv",
                        "--results_csv", "/work/out.csv"], check=True, capture_output=True)
        res = {row["degraded"]: float(row["moslqo"]) for row in csv.DictReader(open(os.path.join(td, "out.csv")))}
    return [res.get(d, np.nan) for _, d in rows]


# ---------------- MERT embeddings ----------------
class Mert:
    def __init__(self, device="cpu"):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.torch, self.dev = torch, device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, trust_remote_code=True)
        self.m = AutoModel.from_pretrained(MERT_ID, trust_remote_code=True).to(device).eval()
        self.sr = self.fe.sampling_rate                              # 24 kHz

    def frames(self, x):
        """Frame embeddings [T, 768]: mean over all hidden layers (MERT's layers carry different musical info)."""
        x24 = resample_poly(x, self.sr, SR).astype(np.float32)
        inp = self.fe(x24, sampling_rate=self.sr, return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            hs = self.m(**inp, output_hidden_states=True).hidden_states
        return self.torch.stack(hs).mean(0)[0].cpu().numpy()

def fad(a, b):
    """Frechet distance between two sets of embeddings (rows = frames)."""
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs, _ = linalg.sqrtm(s1 @ s2, disp=False); cs = cs.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


# ---------------- summaries ----------------
def ci(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0: return None
    h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return {"mean": round(float(v.mean()), 3), "ci95": round(float(h), 3), "n": int(len(v))}

def improvements(items, groups):
    """mean improvement (enhanced - noisy) per metric for each named group of items; MERT-MSE: noisy - enhanced."""
    out = {}
    for name, sel in groups.items():
        rows = [it for it in items if sel(it)]
        if not rows: continue
        g = {}
        for m in ("sisdr", "sdr", "pesq", "visqol"):
            if any(np.isfinite(it.get(f"{m}_e", np.nan)) for it in rows):
                g[m] = ci([it[f"{m}_e"] - it[f"{m}_n"] for it in rows])
        if any(np.isfinite(it.get("mert_e", np.nan)) for it in rows):
            g["mert_mse_reduction"] = ci([it["mert_n"] - it["mert_e"] for it in rows])
        out[name] = g
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("n", nargs="?", type=int, default=120)
    ap.add_argument("--no-visqol", action="store_true"); ap.add_argument("--no-mert", action="store_true")
    ap.add_argument("--mert-device", default="cpu"); ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--split", default="test", choices=["test", "dev"],
                    help="vocals to score: held-out test (default) or dev (for checking the pipeline without the test set)")
    a = ap.parse_args(); n = a.n

    D.CLEAN_W = {"own": 0.5, "vocalset": 0.5, "speech": 0.0}
    ds = D.Mixtures("dev", n, seed=777)
    ds.clean = {"own": sorted(glob.glob(os.path.join(D.DATA, "own_vocals", a.split, "*.wav"))),
                "vocalset": sorted(glob.glob(os.path.join(D.DATA, "vocalset", a.split, "*.wav"))), "speech": []}
    ds.kinds = [k for k in ("own", "vocalset") if ds.clean[k]]; ds.w = np.ones(len(ds.kinds)) / len(ds.kinds)

    R = {k: [] for k in ("sisdr_n", "sisdr_e", "pesq_n", "pesq_e", "passthru")}
    items, sigs = [], []                                          # sigs: (clean, noisy, enhanced) for ViSQOL/MERT
    for i in range(n):
        x, c = (t.numpy() for t in ds[i]); y = process_file(x); meta = dict(ds.last_meta)
        if np.allclose(x, c):
            R["passthru"].append(si_sdr(c, y)); continue
        it = {"i": i, "kind": meta["kind"], "tags": "+".join(sorted(set(meta["tags"]))), "snr_db": meta["snr_db"],
              "sisdr_n": si_sdr(c, x), "sisdr_e": si_sdr(c, y), "pesq_n": p16(c, x), "pesq_e": p16(c, y),
              "sdr_n": sdr(c, x), "sdr_e": sdr(c, y)}
        for k in ("sisdr_n", "sisdr_e", "pesq_n", "pesq_e"): R[k].append(it[k])
        items.append(it); sigs.append((c, x, y))
        if (i + 1) % 25 == 0: print(f"  {i + 1}/{n} items", flush=True)

    norm = [tuple(lufs_norm(s) for s in trio) for trio in sigs]
    if not a.no_visqol:
        print("ViSQOL (audio mode, Docker)...", flush=True)
        try:
            v = visqol_batch([(c, x) for c, x, _ in norm] + [(c, y) for c, _, y in norm])
            for k, it in enumerate(items): it["visqol_n"], it["visqol_e"] = v[k], v[k + len(items)]
        except Exception as e: print(f"  ViSQOL skipped: {e}")
    fad_res = {}
    if not a.no_mert:
        print(f"MERT embeddings ({MERT_ID}, {a.mert_device})...", flush=True)
        try:
            mert = Mert(a.mert_device); E = {"clean": [], "noisy": [], "enh": []}
            for it, (c, x, y) in zip(items, norm):
                ec, ex, ey = mert.frames(c), mert.frames(x), mert.frames(y)
                it["mert_n"], it["mert_e"] = float(((ex - ec) ** 2).mean()), float(((ey - ec) ** 2).mean())
                E["clean"].append(ec); E["noisy"].append(ex); E["enh"].append(ey)
            C = np.concatenate(E["clean"])
            fad_res = {"noisy": fad(np.concatenate(E["noisy"]), C), "enhanced": fad(np.concatenate(E["enh"]), C)}
        except Exception as e: print(f"  MERT skipped: {e}")

    # ---- original summary (unchanged keys) ----
    m = {k: float(np.nanmean(v)) if v else None for k, v in R.items()}
    print(f"{'held-out' if a.split == 'test' else 'DEV (pipeline check, not the test set)'} singing ({n} items: {', '.join(ds.kinds)})")
    print(f"SI-SDR   noisy {m['sisdr_n']:.2f} dB -> enhanced {m['sisdr_e']:.2f} dB  ({m['sisdr_e'] - m['sisdr_n']:+.2f})")
    print(f"PESQ-WB  noisy {m['pesq_n']:.3f} -> enhanced {m['pesq_e']:.3f}  ({m['pesq_e'] - m['pesq_n']:+.3f})")
    if m["passthru"] is not None: print(f"clean pass-through SI-SDR {m['passthru']:.1f} dB  (higher = clean vocals left alone)")

    # ---- music metrics ----
    music = {}
    for key, label, unit in (("sdr", "SDR (BSS-Eval)", " dB"), ("visqol", "ViSQOL audio", ""), ("mert", "MERT-MSE", "")):
        if any(np.isfinite(it.get(f"{key}_e", np.nan)) for it in items):
            nn, ee = (float(np.nanmean([it.get(f"{key}_{s}", np.nan) for it in items])) for s in "ne")
            music[key] = {"noisy": round(nn, 4), "enhanced": round(ee, 4)}
            print(f"{label:<15} noisy {nn:.3f}{unit} -> enhanced {ee:.3f}{unit}  ({ee - nn:+.3f}{'  lower is better' if key == 'mert' else ''})")
    if fad_res:
        music["fad_mert"] = {k: round(v, 4) for k, v in fad_res.items()}
        print(f"FAD-MERT        noisy {fad_res['noisy']:.3f} -> enhanced {fad_res['enhanced']:.3f}  (lower is better)")

    groups = {"all": lambda it: True}
    groups.update({f"snr_{lo:g}-{min(hi, 40):g}dB": (lambda it, lo=lo, hi=hi: lo <= it["snr_db"] < hi) for lo, hi in SNR_BANDS})
    groups.update({f"noise_{t}": (lambda it, t=t: t in it["tags"].split("+")) for t in ("dns_noise", "bleed", "hiss", "hum")})
    groups.update({f"source_{k}": (lambda it, k=k: it["kind"] == k) for k in ds.kinds})
    br = improvements(items, groups)
    print("\nimprovement (enhanced - noisy), mean +/- 95% CI   [MERT: reduction in MSE]")
    for name, g in br.items():
        dp = lambda k: 4 if k.startswith("mert") else 2          # MERT-MSE values are ~0.01, so show more digits
        parts = [f"{k} {v['mean']:+.{dp(k)}f}+/-{v['ci95']:.{dp(k)}f}" for k, v in g.items() if v]
        print(f"  {name:<18} n={next(iter(g.values()))['n']:<4} " + "  ".join(parts))

    m["music"], m["breakdown"] = music, br
    json.dump(m, open(a.out, "w"), indent=1)
    cols = ["i", "kind", "tags", "snr_db", "sisdr_n", "sisdr_e", "sdr_n", "sdr_e", "pesq_n", "pesq_e",
            "visqol_n", "visqol_e", "mert_n", "mert_e"]
    with open(os.path.splitext(a.out)[0] + "_items.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(items)
    print(f"\nwrote {a.out} and {os.path.splitext(a.out)[0]}_items.csv")


if __name__ == "__main__":
    main()
