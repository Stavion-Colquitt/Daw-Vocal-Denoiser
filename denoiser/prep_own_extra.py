"""v1c data: harvest more of the artist's own vocal takes from his Ableton sessions (the folder in DAW_MUSIC).

Every Live set (.als, gzipped XML) is read to find which recorded takes (Samples\\Recorded) each set uses and on
which track. A take is used when:
  - at least one set references it (unreferenced leftovers are skipped: we can't tell what song they belong to)
  - NO held-out set references it (Mad = dev, Say Dat Name = test, incl. masters and the early "say dat shit" set),
    so the dev/test songs can't leak into training
  - it sits on a vocal track: "N-Audio" (his vocal-take tracks), LV / BGV / Lead / Front vocal; anything else
    (sound effects, bounces, instruments) and the instrument-only trumpet project are skipped
Takes are de-duplicated, silence-squeezed the same way as prep_data.py, and written to
<DAW_DATA>/own_vocals_extra/train (originals are only read). A manifest CSV lists every source.
    py -3.12 prep_own_extra.py --dry-run     # report only
    py -3.12 prep_own_extra.py
"""
import os, re, sys, csv, gzip, glob, hashlib, xml.etree.ElementTree as ET
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_data import load_mono48, squeeze_silence, save, SR

MUSIC = os.environ.get("DAW_MUSIC", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ableton-music"))   # your Ableton projects
OUT = os.path.join(os.environ.get("DAW_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "daw-denoiser")), "own_vocals_extra", "train")
HELD_OUT = re.compile(r"^(mad|say dat|sdn)", re.I)
SKIP_PROJECTS = re.compile(r"trumpet project|\\corpus\\|\\_batch", re.I)
VOCAL_TRACK = re.compile(r"^(\d+-Audio\b|LV ?\d*$|BGV ?\d*$|BG Vocal|Lead Vocal|Front vocal|.*\bvocals?\b|.*\bvox\b)", re.I)


def als_refs(path):
    """yield (track_name, absolute_file) for every audio clip in a Live set"""
    try:
        root = ET.fromstring(gzip.open(path).read())
    except Exception as e:
        print(f"  skip unreadable set {path}: {e}"); return
    base = os.path.dirname(path)
    for tr in root.iter("AudioTrack"):
        nm = tr.find("Name/EffectiveName"); name = nm.get("Value") if nm is not None else ""
        for clip in tr.iter("AudioClip"):
            fr = clip.find("SampleRef/FileRef")
            if fr is None: continue
            p = fr.find("Path"); rel = fr.find("RelativePath")
            cand = []
            if p is not None and p.get("Value"): cand.append(p.get("Value"))
            if rel is not None and rel.get("Value"): cand.append(os.path.join(base, rel.get("Value")))
            for c in cand:
                c = os.path.normpath(c.replace("/", os.sep))
                if os.path.exists(c): yield name, c; break


def main(dry):
    sets = [p for p in glob.glob(os.path.join(MUSIC, "**", "*.als"), recursive=True) if "Backup" not in p]
    uses = defaultdict(set)                       # file -> {(set, track)}
    for s in sets:
        for track, f in als_refs(s):
            uses[os.path.normcase(f)].add((os.path.splitext(os.path.basename(s))[0], track))
    recorded = [f for f in uses if "\\samples\\recorded\\" in f and f.endswith((".wav", ".aif", ".aiff"))]
    keep, why = [], defaultdict(int)
    for f in sorted(recorded):
        u = uses[f]
        if any(HELD_OUT.match(st) for st, _ in u): why["held-out song"] += 1; continue
        if SKIP_PROJECTS.search(f): why["instrument/test project"] += 1; continue
        if not any(VOCAL_TRACK.match(t or "") for _, t in u): why["not a vocal track"] += 1; continue
        keep.append(f)
    print(f"{len(sets)} sets, {len(uses)} referenced files, {len(recorded)} recorded takes -> keep {len(keep)}; skipped {dict(why)}")
    seen, tot, rows = set(), 0.0, []
    for f in keep:
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h in seen: continue
        seen.add(h)
        songs = sorted({st for st, _ in uses[f]}); tracks = sorted({t for _, t in uses[f]})
        if dry:
            rows.append([f, "|".join(songs), "|".join(tracks), ""]); continue
        try:
            x = load_mono48(f)
            y = squeeze_silence(x) if len(x) >= SR else x[:0]    # takes under 1 s are useless (and trip the squeeze)
        except Exception as e:
            print(f"  skip {f}: {e}", flush=True); continue
        if len(y) == 0: continue
        name = re.sub(r"[^A-Za-z0-9]+", "_", f"{songs[0]}_{os.path.basename(f)[:-4]}").strip("_")[:120]
        save(y, os.path.join(OUT, name + ".wav")); tot += len(y) / SR
        rows.append([f, "|".join(songs), "|".join(tracks), f"{len(y) / SR:.1f}"])
    by_song = defaultdict(int)
    for r in rows: by_song[r[1].split("|")[0]] += 1
    print("takes per song:", dict(sorted(by_song.items(), key=lambda kv: -kv[1])))
    man = os.path.join(os.path.dirname(OUT), "manifest_dry.csv" if dry else "manifest.csv")
    os.makedirs(os.path.dirname(man), exist_ok=True)
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["source", "sets", "tracks", "seconds_after_squeeze"]); w.writerows(rows)
    print(f"{len(rows)} unique takes" + ("" if dry else f", {tot / 60:.1f} minutes of singing") + f"; manifest: {man}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
