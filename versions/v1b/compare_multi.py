"""Run compare.py with a per-model streaming config: label=path.onnx@HIDDEN,LOOKAHEAD (v1b: 384,0; v1d: 256,1)."""
import sys, stream as St, compare as C
CFG, args = {}, []
for a in sys.argv[1:]:
    if "=" in a and "@" in a:
        lab, rest = a.split("=", 1); p, cfg = rest.rsplit("@", 1); h, la = cfg.split(","); CFG[p] = (int(h), int(la)); args.append(f"{lab}={p}")
    else: args.append(a)
def pf(x, path=None, **kw):
    h, la = CFG.get(path, (256, 0)); St.LOOKAHEAD = la
    return St.process_file(x, path=path, hidden=h, **kw)
C.process_file = pf
sys.argv = [sys.argv[0]] + args
C.main()