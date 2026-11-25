"""On-the-fly mixtures for the DAW vocal denoiser (48 kHz).

clean  : own vocal takes (35%) | VocalSet singing (35%) | DNS Challenge clean speech (30%)
noise  : any mix of DNS noise, headphone bleed (own instrumentals, high-passed), hiss, mains hum
room   : 30% of examples get a synthetic room (same reverb on input AND target: the model removes noise,
         not the room)
SNR 0..40 dB (tracking noise is usually quiet), 10% clean-only examples (the model must leave clean vocals alone).
Fixed train/dev split by file; dev items are generated from a fixed seed so every eval sees the same mixes.
"""
import os, glob, random, zlib, numpy as np, soundfile as sf, torch
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly
from torch.utils.data import Dataset, DataLoader

SR = 48000
SEG = 4 * SR          # long segments let the GRU learn context (Xia et al. 2020: >= 5 s best; 4 s fits GPU memory)
DATA = os.environ.get("DAW_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "daw-denoiser"))   # built by prep_data.py
DNS = os.environ.get("DNS_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dns"))   # clean_fullband/ + noise_fullband/
SNR_MIN, SNR_MAX, P_CLEAN_ONLY, P_ROOM = 0.0, 40.0, 0.10, 0.30
CLEAN_W = {"own": 0.35, "vocalset": 0.35, "speech": 0.30}

# --- Leakage guard (audit of 2026-11-25, see denoiser-report/LEAKAGE_AUDIT_2026-11-25.md) -----------------------
# The DNS clean pool carries a second, unsplit copy of VocalSet (folder VocalSet_48kHz_mono, 3,613 files, names
# "vocalset_<singer>_<take>_48kHz.wav", re-authored at 3x speed). All 20 singers are in it, including the held-out
# female9 and male11. Singing already enters training through vocalset/train, which is split by singer, so every
# VocalSet-derived file is filtered out of the DNS speech branch here. Nothing is moved or deleted: the files stay
# on disk, they are simply never drawn as a clean target. This is a filter, not a data change.
VOCALSET_SINGERS = tuple(f"female{i}" for i in range(1, 10)) + tuple(f"male{i}" for i in range(1, 12))
VOCALSET_HOLDOUT = ("female9", "male11")       # the held-out test singers
VOCALSET_DEV = ("female8", "male10")           # the VocalSet dev singers


_VOCALSET_NAMES = None


def _vocalset_names():
    """Basenames of every file in the DNS copy of VocalSet (cached). Used as an independent overlap check."""
    global _VOCALSET_NAMES
    if _VOCALSET_NAMES is None:
        pat = os.path.join(DNS, "clean_fullband", "**", "VocalSet*", "*.wav")
        _VOCALSET_NAMES = {os.path.basename(f).lower() for f in glob.glob(pat, recursive=True)}
    return _VOCALSET_NAMES


def _is_vocalset_derived(path):
    """True if a DNS clean file is a VocalSet copy: named vocalset_* or carrying any VocalSet singer token."""
    n = os.path.basename(path).lower()
    return "vocalset" in n or any(s in n for s in VOCALSET_SINGERS)


def _speech_pool(split):
    """DNS clean speech for this split, with every VocalSet-derived file removed (leakage audit 2026-11-25)."""
    files = [f for f in glob.glob(os.path.join(DNS, "clean_fullband", "**", "*.wav"), recursive=True)
             if _is_dev(f) == (split != "train")]
    kept = [f for f in files if not _is_vocalset_derived(f)]
    names = _vocalset_names()     # fail loud, independent overlap check: no kept speech file may share a basename
    clash = [f for f in kept if os.path.basename(f).lower() in names]      # with the DNS copy of VocalSet
    if clash:
        raise RuntimeError(f"{len(clash)} speech file(s) for {split!r} are still VocalSet copies, e.g. "
                           f"{clash[:3]}. Refusing to train on leaked data (see LEAKAGE_AUDIT_2026-11-25.md).")
    if not kept:
        raise RuntimeError(f"DNS speech pool for {split!r} is empty after the VocalSet filter (DNS_DATA={DNS}).")
    return kept


def _is_dev(path, pct=5): return zlib.crc32(os.path.basename(path).encode()) % 100 < pct

def pools(split):
    g = lambda pat: sorted(glob.glob(pat, recursive=True))
    sp = "train" if split == "train" else "dev"
    speech = _speech_pool(split)                    # DNS clean speech, VocalSet copies filtered out (2026-11-25 audit)
    noise = [f for f in g(os.path.join(DNS, "noise_fullband", "**", "*.wav")) if _is_dev(f) == (split != "train")]
    own = g(os.path.join(DATA, "own_vocals", sp, "*.wav"))
    if split == "train" and os.environ.get("DAW_OWN_EXTRA"):          # v1c: more of my own takes (train only)
        own += g(os.path.join(DATA, os.environ["DAW_OWN_EXTRA"], "train", "*.wav"))
    clean = {"own": own,
             "vocalset": g(os.path.join(DATA, "vocalset", sp, "*.wav")), "speech": speech}
    return clean, noise, g(os.path.join(DATA, "bleed", "*.wav"))


def read_crop(path, n, rng, min_rms=1e-3, tries=6):
    """Random n-sample mono crop at 48 kHz (loops short files); skips near-silent crops."""
    info = sf.info(path); sr = info.samplerate; need = int(np.ceil(n * sr / SR))
    for _ in range(tries):
        if info.frames > need:
            s = rng.randint(0, info.frames - need); x, _ = sf.read(path, start=s, frames=need, dtype="float32", always_2d=True)
        else:
            x, _ = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            g = np.gcd(SR, sr); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        if len(x) < n: x = np.tile(x, int(np.ceil(n / max(len(x), 1))))
        x = x[:n]
        if np.sqrt(np.mean(x ** 2)) > min_rms: return x
    return x


def rms(x): return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def synth_rir(rng):
    t60 = rng.uniform(0.15, 0.7); L = int(t60 * SR)
    t = np.arange(L) / SR
    h = np.random.default_rng(rng.randint(0, 2 ** 31)).standard_normal(L) * np.exp(-6.9 * t / t60)
    h[: int(0.002 * SR)] = 0.0                     # gap before the first reflections
    h *= 10 ** (rng.uniform(-18, -6) / 20) / (np.abs(h).max() + 1e-9)
    h[0] = 1.0                                      # direct path
    return h.astype(np.float32)


def colored_noise(n, rng, kind):
    w = np.random.default_rng(rng.randint(0, 2 ** 31)).standard_normal(n).astype(np.float32)
    if kind == "pink":
        W = np.fft.rfft(w); f = np.arange(len(W)); f[0] = 1; w = np.fft.irfft(W / np.sqrt(f), n).astype(np.float32)
    return w


def random_eq(x, rng):
    """Mild random tilt (low/high shelf, about +-4 dB): different mics, preamps and singers' tone."""
    for kind, lo, hi in (("lp", 150, 600), ("hp", 3000, 9000)):
        if rng.random() < 0.5:
            x = x + rng.uniform(-0.35, 0.6) * sosfilt(butter(1, rng.uniform(lo, hi), kind, fs=SR, output="sos"), x)
    return x.astype(np.float32)


def hum(n, rng):
    f0 = rng.choice([50.0, 60.0]); t = np.arange(n) / SR
    return sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) for k in range(1, 8)).astype(np.float32)


class Mixtures(Dataset):
    def __init__(self, split, length, seed=None):
        self.clean, self.noise, self.bleed = pools(split)
        self.kinds = [k for k in CLEAN_W if self.clean[k]]
        self.w = np.array([CLEAN_W[k] for k in self.kinds]); self.w /= self.w.sum()
        self.length, self.seed = length, seed

    def __len__(self): return self.length

    def __getitem__(self, i):
        rng = random.Random(self.seed * 100003 + i) if self.seed is not None else random.Random()
        kind = self.kinds[int(np.searchsorted(np.cumsum(self.w), rng.random()))]
        clean = random_eq(read_crop(rng.choice(self.clean[kind]), SEG, rng), rng)
        if rng.random() < P_ROOM:
            clean = fftconvolve(clean, synth_rir(rng))[:SEG].astype(np.float32)
        clean *= 10 ** (rng.uniform(-32, -12) / 20) / rms(clean)      # random vocal level (RMS dBFS)
        if rng.random() < P_CLEAN_ONLY:
            self.last_meta = {"kind": kind, "tags": ["clean"], "snr_db": None}
            return torch.from_numpy(clean.copy()), torch.from_numpy(clean.copy())
        parts, tags = [], []   # tags: what was mixed in (for per-noise-type eval)
        if self.noise and rng.random() < 0.65: parts.append(read_crop(rng.choice(self.noise), SEG, rng, 1e-4)); tags.append("dns_noise")
        if self.bleed and rng.random() < 0.35:
            b = read_crop(rng.choice(self.bleed), SEG, rng, 1e-4)
            b = sosfilt(butter(2, rng.uniform(200, 900), "hp", fs=SR, output="sos"), b)       # headphones leak mids/highs
            parts.append(b.astype(np.float32)); tags.append("bleed")
        if rng.random() < 0.30: parts.append(colored_noise(SEG, rng, rng.choice(["white", "pink"]))); tags.append("hiss")
        if rng.random() < 0.15: parts.append(hum(SEG, rng)); tags.append("hum")
        if not parts: parts.append(colored_noise(SEG, rng, "pink")); tags.append("hiss")
        noise = sum(p / rms(p) * 10 ** (rng.uniform(-6, 0) / 20) for p in parts).astype(np.float32)
        snr = rng.uniform(SNR_MIN, SNR_MAX)
        noise *= rms(clean) / rms(noise) / 10 ** (snr / 20)
        self.last_meta = {"kind": kind, "tags": tags, "snr_db": snr}
        noisy = clean + noise
        pk = np.abs(noisy).max()
        if pk > 0.99: noisy, clean = noisy * (0.99 / pk), clean * (0.99 / pk)
        return torch.from_numpy(noisy.astype(np.float32)), torch.from_numpy(clean.astype(np.float32))


def loaders(batch=32, steps_per_epoch=1000, dev_items=256, workers=6):
    tr = DataLoader(Mixtures("train", batch * steps_per_epoch), batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, persistent_workers=workers > 0, drop_last=True)
    dv = DataLoader(Mixtures("dev", dev_items, seed=1234), batch_size=batch, shuffle=False, num_workers=0)
    return tr, dv


if __name__ == "__main__":
    for s in ("train", "dev"):
        c, n, b = pools(s); print(s, {k: len(v) for k, v in c.items()}, "noise", len(n), "bleed", len(b))
    ds = Mixtures("train", 4); x, y = ds[0]; print(x.shape, y.shape, float(x.abs().max()))
