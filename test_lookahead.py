# Checks that DAW_LOOKAHEAD=N lets output sample block t depend on input up to N hops (5 ms each) ahead, and no further.
import os, sys, importlib, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.manual_seed(0)
for la in (0, 1):
    os.environ["DAW_LOOKAHEAD"] = str(la); import model; importlib.reload(model)
    from model import StreamDenoiser, enhance, M, SR
    net = StreamDenoiser(hidden=64).eval(); x = torch.randn(1, SR // 4)
    t0 = 20 * M                                    # cut point: output before t0 vs input after t0
    with torch.no_grad():
        y = enhance(net, x)
        res = []
        for k in range(0, 3):                      # perturb input starting k hops after t0
            x2 = x.clone(); x2[:, t0 + k * M:] += 1.0
            res.append((enhance(net, x2)[:, :t0] - y[:, :t0]).abs().max().item())
    print(f"lookahead={la}: output before t0 changes when input changes at t0+{{0,1,2}} hops: "
          + ", ".join(f"{r:.1e}" for r in res))
