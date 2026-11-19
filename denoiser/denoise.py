"""DAW Vocal Denoiser in a container: CPU only (ONNX Runtime), streaming engine (10 ms latency, +5 ms per lookahead frame).

  CLI (default):  denoise.py IN.wav OUT.wav [--floor 0.0]      relative paths are under /data (the mounted folder)
  Server:         denoise.py serve [--port 8000]                POST a WAV/FLAC to /denoise, get the cleaned WAV back
                                                                GET /health -> {"ok": true, ...}
Any sample rate and channel count: audio is resampled to 48 kHz, each channel is denoised separately, then it is
resampled back to the original rate. Output is 24-bit PCM WAV.
"""
import argparse, io, json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np, soundfile as sf, soxr
from stream import SR, process_file

MODEL = "DAW Vocal Denoiser " + os.environ.get("DAW_VERSION", "")
MAX_BYTES = 200 * 1024 * 1024          # server upload limit (about 20 min of 48 kHz stereo 24-bit)


def denoise_array(x, sr, floor=0.0):
    """x: float32 [n] or [n, ch] at any rate -> same shape and rate, denoised."""
    x = np.asarray(x, np.float32); mono = x.ndim == 1
    if mono: x = x[:, None]
    y48 = soxr.resample(x, sr, SR) if sr != SR else x
    out = np.stack([process_file(y48[:, c], mask_floor=floor) for c in range(y48.shape[1])], axis=1)
    if sr != SR: out = soxr.resample(out, SR, sr)
    out = out[: len(x)]
    return out[:, 0] if mono else out


def run_cli(a):
    x, sr = sf.read(a.input, dtype="float32", always_2d=False)
    t = time.time(); y = denoise_array(x, sr, a.floor); dt = time.time() - t
    sf.write(a.output, y, sr, subtype="PCM_24")
    secs = len(x) / sr
    print(f"{MODEL}: {os.path.basename(a.input)} ({secs:.1f} s, {sr} Hz, {1 if x.ndim == 1 else x.shape[1]} ch) "
          f"-> {os.path.basename(a.output)} in {dt:.1f} s ({secs / max(dt, 1e-9):.0f}x real time)")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, json.dumps({"ok": True, "model": MODEL, "sample_rate": SR, "latency_ms": 10}).encode(),
                       "application/json")
        else: self._send(404, b'{"error": "not found"}', "application/json")

    def do_POST(self):
        if not self.path.startswith("/denoise"): return self._send(404, b'{"error": "not found"}', "application/json")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BYTES: return self._send(413, b'{"error": "empty or too large"}', "application/json")
        try:
            x, sr = sf.read(io.BytesIO(self.rfile.read(n)), dtype="float32", always_2d=False)
            buf = io.BytesIO(); sf.write(buf, denoise_array(x, sr), sr, format="WAV", subtype="PCM_24")
            self._send(200, buf.getvalue(), "audio/wav")
        except Exception as e:
            self._send(400, json.dumps({"error": f"could not process audio: {e}"}).encode(), "application/json")

    def log_message(self, fmt, *args): sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        p = argparse.ArgumentParser(prog="denoise.py serve"); p.add_argument("serve"); p.add_argument("--port", type=int, default=8000)
        a = p.parse_args(); print(f"{MODEL} listening on :{a.port}  (POST /denoise, GET /health)", flush=True)
        ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()
    p = argparse.ArgumentParser(description=f"{MODEL}: denoise a vocal recording (any rate, mono or stereo).")
    p.add_argument("input"); p.add_argument("output")
    p.add_argument("--floor", type=float, default=0.0, help="minimum mask value, 0-1 (higher = gentler)")
    run_cli(p.parse_args())


if __name__ == "__main__":
    main()
