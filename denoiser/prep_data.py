"""Build the training pool for the DAW vocal denoiser (48 kHz mono float32 WAVs).

Sources (all either the artist's own recordings or openly licensed):
  own_vocals : individual vocal takes from the artist's own sessions (read-only; originals never touched)
  vocalset   : VocalSet (CC BY 4.0, Wilkins et al., ISMIR 2018), resampled 44.1 -> 48 kHz
  bleed      : the artist's own instrumentals, used to simulate headphone bleed (songs with Cymatics
               sample tracks are skipped on purpose)
DNS Challenge clean speech + noise (CC BY 4.0) are read in place by dataset.py.

Long silences are squeezed out of takes (most session takes are mostly silence).
    py -3.12 prep_data.py own | vocalset | bleed | all
"""
import os, re, sys, glob, numpy as np, soundfile as sf
from scipy.signal import resample_poly

SR = 48000
OUT = os.environ.get("DAW_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "daw-denoiser"))      # training pool written here
CORPUS = os.environ.get("DAW_CORPUS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corpus"))  # your own sessions: <song>/*.wav takes
# songs whose individual vocal takes are exported in the corpus. "Say Dat Name" is held out for TEST.
OWN_SONGS = {"Do It": "train", "Wanna Sleep Stems": "train", "Never Beat me": "train", "If you leave": "train",
             "I don't fall in love": "train", "I Don't Like It": "train", "I don't like you": "train",
             "Rap Song": "train", "Mad": "dev", "Say Dat Name": "test"}
# individual takes only: "N-Audio" tracks (vocal takes in this workflow) and numbered LV / BGV / BG Vocal / Front vocal
TAKE_RE = re.compile(r"(\d+-Audio( \d+)?( \[[^\]]+\])?|LV \d+|BGV \d+|BG Vocal \d+|Front vocal \d+)\.wav$", re.I)
# instrumentals for bleed: only songs with no Cymatics sample tracks
BLEED_SONGS = {"Wanna Sleep Stems": "instru", "Never Beat me": "Instra", "If you leave": "Instra",
               "Say Dat Name": "Instruments", "I don't fall in love": "Instra"}


def load_mono48(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(SR, sr); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def squeeze_silence(x, thr_db=-50.0, keep_s=0.15, min_len_s=1.0):
    """Keep active regions (20 ms RMS above thr_db, relative to full scale) plus keep_s of context; join them."""
    hop = SR // 50; n = len(x) // hop
    if n == 0: return np.zeros(0, np.float32)
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1) + 1e-12)
    act = 20 * np.log10(rms) > thr_db
    k = int(keep_s * 50)
    act = np.convolve(act.astype(float), np.ones(2 * k + 1), "same") > 0
    keep = np.repeat(act, hop)
    y = x[:n * hop][keep]
    return y if len(y) >= min_len_s * SR else np.zeros(0, np.float32)


def save(y, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pk = np.abs(y).max()
    if pk > 0.99: y = y * (0.99 / pk)
    sf.write(path, y, SR, subtype="FLOAT")


def prep_own():
    tot = {"train": 0.0, "dev": 0.0, "test": 0.0}
    for song, split in OWN_SONGS.items():
        for p in sorted(glob.glob(os.path.join(CORPUS, song, "*.wav"))):
            if not TAKE_RE.search(os.path.basename(p)): continue
            y = squeeze_silence(load_mono48(p))
            if len(y) == 0: continue
            name = re.sub(r"[^A-Za-z0-9]+", "_", f"{song}_{os.path.basename(p)[:-4]}").strip("_")
            save(y, os.path.join(OUT, "own_vocals", split, name + ".wav")); tot[split] += len(y) / SR
            print(f"  {split:5s} {len(y) / SR:6.1f}s  {song} / {os.path.basename(p)}", flush=True)
    print("own vocals (minutes):", {k: round(v / 60, 1) for k, v in tot.items()})


def prep_vocalset():
    root = os.path.join(OUT, "raw", "FULL")          # VocalSet.zip extracts to FULL/<singer>/<technique>/...
    files = glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
    # hold out 2 female + 2 male singers (by VocalSet singer id) for dev/test
    hold = {"female9": "test", "male11": "test", "female8": "dev", "male10": "dev"}
    tot = {"train": 0.0, "dev": 0.0, "test": 0.0}
    for p in files:
        rel = os.path.relpath(p, root).replace("\\", "/")
        singer = next((s for s in rel.split("/") if re.fullmatch(r"(fe)?male\d+", s)), None)
        if singer is None: continue
        split = hold.get(singer, "train")
        y = squeeze_silence(load_mono48(p))
        if len(y) == 0: continue
        name = re.sub(r"[^A-Za-z0-9]+", "_", rel[:-4]).strip("_")
        save(y, os.path.join(OUT, "vocalset", split, name + ".wav")); tot[split] += len(y) / SR
    print(f"vocalset: {len(files)} files ->", {k: round(v / 60, 1) for k, v in tot.items()}, "minutes")


def prep_bleed():
    for song, stem in BLEED_SONGS.items():
        hits = [p for p in glob.glob(os.path.join(CORPUS, song, "*.wav")) if p.lower().endswith(f" {stem.lower()}.wav")]
        for p in hits:
            y = squeeze_silence(load_mono48(p), thr_db=-45)
            name = re.sub(r"[^A-Za-z0-9]+", "_", f"{song}_{stem}").strip("_")
            save(y, os.path.join(OUT, "bleed", name + ".wav")); print(f"  bleed {len(y) / SR:6.1f}s  {song}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("own", "all"): prep_own()
    if what in ("bleed", "all"): prep_bleed()
    if what in ("vocalset", "all"): prep_vocalset()
