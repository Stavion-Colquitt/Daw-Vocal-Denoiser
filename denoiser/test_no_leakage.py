"""Repo test: the DNS speech branch must contain no VocalSet-derived file (leakage audit 2026-11-25).

Re-runs the overlap check that found the leak. The DNS clean pool ships a second, unsplit copy of VocalSet
(folder VocalSet_48kHz_mono, names vocalset_<singer>_<take>_48kHz.wav). The held-out test singers (female9, male11)
and the VocalSet dev singers (female8, male10) must never reach the speech branch, and the copies must still be on
disk: the fix is a filter in dataset.py, not a data change.

    py -3.12 denoiser\test_no_leakage.py        # exit code 0 = clean, 1 = leak

Reads file names only: no audio, no GPU, a few seconds.
"""
import os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D


def main():
    fails, notes = [], []
    dns = D.DNS
    print(f"DNS_DATA = {dns}")
    if not os.path.isdir(dns):
        print("FAIL: DNS_DATA folder not found")
        return 1

    copies = sorted(glob.glob(os.path.join(dns, "clean_fullband", "**", "VocalSet*", "*.wav"), recursive=True))
    print(f"VocalSet copies still on disk (nothing deleted): {len(copies)}")
    if not copies:
        fails.append("no VocalSet copy found on disk, so the overlap check would be meaningless (check DNS_DATA)")
    copy_names = {os.path.basename(f).lower() for f in copies}

    all_clean = glob.glob(os.path.join(dns, "clean_fullband", "**", "*.wav"), recursive=True)
    for split in ("train", "dev"):
        raw = [f for f in all_clean if D._is_dev(f) == (split != "train")]
        speech = D._speech_pool(split)                      # raises RuntimeError itself if the filter failed
        dropped = len(raw) - len(speech)
        print(f"{split:5s}: DNS speech {len(raw)} -> {len(speech)} after the VocalSet filter ({dropped} dropped)")
        leak = [f for f in speech if D._is_vocalset_derived(f)
                or os.path.basename(f).lower() in copy_names
                or any(s in os.path.basename(f).lower() for s in D.VOCALSET_HOLDOUT)]
        if leak:
            fails.append(f"{split}: {len(leak)} VocalSet-derived file(s) in the speech branch, e.g. {leak[:3]}")
        if dropped == 0:
            notes.append(f"{split}: nothing dropped, is the DNS VocalSet copy still where the audit found it?")

    # the singing branch must stay split by singer: held-out singers in dev only, never in train
    cl_tr, _, _ = D.pools("train")
    cl_dv, _, _ = D.pools("dev")
    for singer in D.VOCALSET_HOLDOUT:
        n_tr = sum(singer in os.path.basename(f).lower() for f in cl_tr["vocalset"])
        n_dv = sum(singer in os.path.basename(f).lower() for f in cl_dv["vocalset"])
        print(f"held-out singer {singer}: {n_tr} file(s) in vocalset/train, {n_dv} in vocalset/dev")
        if n_tr:
            fails.append(f"held-out singer {singer} appears in vocalset/train ({n_tr} files)")
        if not n_dv:
            notes.append(f"held-out singer {singer} has no vocalset/dev files")

    for n in notes:
        print(f"note: {n}")
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        print("FAIL: leakage check")
        return 1
    print("PASS: no VocalSet-derived file reaches the DNS speech branch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
