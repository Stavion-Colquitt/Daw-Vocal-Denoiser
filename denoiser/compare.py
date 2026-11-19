"""Compare my own model versions on IDENTICAL clips (same fixed-seed mixtures, same metrics as eval.py).

    py -3.12 compare.py v1=path/to/v1.onnx v1c=path/to/v1c.onnx [...] [--n 120] [--split dev|test] [--no-visqol] [--no-mert]

Every model runs through the streaming engine (what a DAW would run). Prints a markdown table (absolute scores, the
noisy input as the first row) plus improvement-over-noisy with 95% CIs, and paired differences between consecutive
versions (same clips, so the comparison is paired). Writes compare_results.json and compare_results.md.
"""
import os, sys, glob, json, argparse, numpy as np
import dataset as D
import eval as E
from stream import process_file


def paired(a, b):
    """mean and 95% CI of b - a over the same items."""
    d = np.asarray(b, float) - np.asarray(a, float); d = d[np.isfinite(d)]
    if len(d) < 2: return None
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d))), int(len(d))


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")                      # Windows consoles default to cp1252
    except Exception: pass
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+", help="label=path.onnx, oldest first")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--no-visqol", action="store_true"); ap.add_argument("--no-mert", action="store_true")
    ap.add_argument("--out", default="compare_results")
    a = ap.parse_args()
    models = [m.split("=", 1) for m in a.models]
    for lab, p in models: assert os.path.exists(p), f"missing {lab}: {p}"

    D.CLEAN_W = {"own": 0.5, "vocalset": 0.5, "speech": 0.0}
    ds = D.Mixtures("dev", a.n, seed=777)
    ds.clean = {"own": sorted(glob.glob(os.path.join(D.DATA, "own_vocals", a.split, "*.wav"))),
                "vocalset": sorted(glob.glob(os.path.join(D.DATA, "vocalset", a.split, "*.wav"))), "speech": []}
    ds.kinds = [k for k in ("own", "vocalset") if ds.clean[k]]; ds.w = np.ones(len(ds.kinds)) / len(ds.kinds)

    clips, passthru = [], {lab: [] for lab, _ in models}
    for i in range(a.n):
        x, c = (t.numpy() for t in ds[i])
        if np.allclose(x, c):
            for lab, p in models: passthru[lab].append(E.si_sdr(c, process_file(x, path=p)))
            continue
        clips.append((c, x, dict(ds.last_meta)))
    print(f"{len(clips)} noisy clips + {a.n - len(clips)} clean pass-through clips ({a.split} split)", flush=True)

    rows = {"noisy": [x for _, x, _ in clips]}
    for lab, p in models:
        rows[lab] = [process_file(x, path=p) for _, x, _ in clips]; print(f"  ran {lab}", flush=True)
    refs = [c for c, _, _ in clips]

    S = {}                                   # S[label][metric] = per-clip list
    for lab, outs in rows.items():
        S[lab] = {"sisdr": [E.si_sdr(c, y) for c, y in zip(refs, outs)], "sdr": [E.sdr(c, y) for c, y in zip(refs, outs)],
                  "pesq": [E.p16(c, y) for c, y in zip(refs, outs)]}
    nref = [E.lufs_norm(c) for c in refs]; nout = {lab: [E.lufs_norm(y) for y in outs] for lab, outs in rows.items()}
    if not a.no_visqol:
        print("ViSQOL...", flush=True)
        labs = list(rows); v = E.visqol_batch([(r, y) for lab in labs for r, y in zip(nref, nout[lab])])
        for k, lab in enumerate(labs): S[lab]["visqol"] = v[k * len(refs):(k + 1) * len(refs)]
    fad = {}
    if not a.no_mert:
        print("MERT...", flush=True)
        mert = E.Mert("cpu"); er = [mert.frames(r) for r in nref]; C = np.concatenate(er)
        for lab in rows:
            ey = [mert.frames(y) for y in nout[lab]]
            S[lab]["mert_mse"] = [float(((b - r) ** 2).mean()) for b, r in zip(ey, er)]
            fad[lab] = E.fad(np.concatenate(ey), C)

    metrics = [m for m in ("sisdr", "sdr", "pesq", "visqol", "mert_mse") if m in S["noisy"]]
    names = {"sisdr": "SI-SDR (dB)", "sdr": "SDR (dB)", "pesq": "PESQ-WB", "visqol": "ViSQOL", "mert_mse": "MERT-MSE ↓"}
    hdr = "| Version | " + " | ".join(names[m] for m in metrics) + (" | FAD-MERT ↓" if fad else "") + " | Pass-through SI-SDR (dB) |"
    lines = [f"Same {len(clips)} {a.split} clips for every version (fixed seed 777), streaming engine.", "", hdr,
             "|" + "---|" * (len(metrics) + (2 if fad else 1) + 1)]
    for lab in rows:
        cells = [f"{np.nanmean(S[lab][m]):.4f}" if m == "mert_mse" else f"{np.nanmean(S[lab][m]):.2f}" for m in metrics]
        if fad: cells.append(f"{fad[lab]:.3f}")
        pt = f"{np.mean(passthru[lab]):.1f}" if lab in passthru and passthru[lab] else "—"
        lines.append(f"| {lab} | " + " | ".join(cells) + f" | {pt} |")
    labs = [lab for lab, _ in models]
    lines += ["", "Paired differences between consecutive versions (newer - older, mean ± 95% CI; for MERT-MSE negative = better):", ""]
    for old, new in zip(labs, labs[1:]):
        parts = []
        for m in metrics:
            r = paired(S[old][m], S[new][m])
            if r: parts.append(f"{names[m]} {r[0]:+.4f} ± {r[1]:.4f}" if m == "mert_mse" else f"{names[m]} {r[0]:+.2f} ± {r[1]:.2f}")
        lines.append(f"- **{new} vs {old}**: " + ", ".join(parts))
    kinds = [meta["kind"] for _, _, meta in clips]
    lines += ["", "By voice (mean score; paired change from the first to the last version, ± 95% CI):", ""]
    for who in sorted(set(kinds)):
        sel = np.array([k == who for k in kinds])
        parts = []
        for m in metrics:
            r = paired(np.array(S[labs[0]][m])[sel], np.array(S[labs[-1]][m])[sel])
            if r: parts.append(f"{names[m]} {r[0]:+.4f} ± {r[1]:.4f}" if m == "mert_mse" else f"{names[m]} {r[0]:+.2f} ± {r[1]:.2f}")
        lines.append(f"- **{who}** ({int(sel.sum())} clips), {labs[-1]} vs {labs[0]}: " + ", ".join(parts))
    md = "\n".join(lines)
    open(a.out + ".md", "w", encoding="utf-8").write(md + "\n")          # save first, print second
    json.dump({"split": a.split, "n_clips": len(clips), "models": dict(models), "fad_mert": fad, "kinds": kinds,
               "mean": {lab: {m: float(np.nanmean(S[lab][m])) for m in metrics} for lab in rows},
               "passthru_sisdr": {k: float(np.mean(v)) if v else None for k, v in passthru.items()},
               "per_clip": {lab: {m: [float(v) for v in S[lab][m]] for m in metrics} for lab in rows}},
              open(a.out + ".json", "w"), indent=1)
    print("\n" + md); print(f"\nwrote {a.out}.md and {a.out}.json")


if __name__ == "__main__":
    main()
