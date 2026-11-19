"""Real-time streaming engine: 240-sample (5 ms) blocks in, 240-sample blocks out, 10 ms algorithmic latency.

Needs only numpy + onnxruntime (no PyTorch). Each call: shift the block into a 1536-sample (32 ms) history, apply the
long analysis window, FFT, run one GRU step (state kept between calls), mask, inverse FFT, apply the short synthesis
window, and overlap-add. The emitted block is the PREVIOUS 5 ms of audio, now complete, so input -> output delay is
2 x 240 samples = 10 ms. Same math as model.analysis/synthesis in the training code (parity-checked).
    eng = StreamEngine(); y_block = eng.process(x_block)   # x_block: float32[240] at 48000 Hz
    y = process_file(x)                                     # whole mono 48000 Hz signal, time-aligned output
"""
import os, numpy as np, onnxruntime as ort

SR, K, M, D_ZEROS = 48000, 1536, 240, 64          # sample rate, analysis length, hop, leading zeros
HIDDEN = int(os.environ.get("DAW_HIDDEN", 384))         # v1b: GRU width
LOOKAHEAD = int(os.environ.get("DAW_LOOKAHEAD", 0))   # v1b: frames of lookahead (1 frame = 5 ms)
ONNX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daw_denoiser.onnx")


def windows():
    """Asymmetric analysis/synthesis pair (Wang et al. 2021): analysis * synthesis = Hann(2M) on the last 2M samples."""
    def hann(n): return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    L = 2 * (K - M - D_ZEROS); hL, h2M = hann(L), hann(2 * M)
    wa = np.zeros(K); ws = np.zeros(K)
    wa[D_ZEROS:K - M] = np.sqrt(hL[: K - M - D_ZEROS])
    wa[K - M:] = np.sqrt(h2M[M:])
    seg = slice(K - 2 * M, K - M)
    ws[seg] = h2M[:M] / np.maximum(wa[seg], 1e-12)
    ws[K - M:] = np.sqrt(h2M[M:])
    return wa, ws


class StreamEngine:
    def __init__(self, path=ONNX, layers=2, hidden=HIDDEN, mask_floor=0.0):
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        wa, ws = windows(); self.wa = wa.astype(np.float32); self.ws = ws[K - 2 * M:].astype(np.float32)
        self.h = np.zeros((layers, 1, hidden), np.float32); self.floor = mask_floor
        self.inbuf = np.zeros(K, np.float32); self.tail = np.zeros(M, np.float32)
        self.la = LOOKAHEAD; self.sq = [np.zeros(K // 2 + 1, np.complex128) for _ in range(self.la)]   # spectra waiting for their mask

    def reset(self):
        self.h[:] = 0; self.inbuf[:] = 0; self.tail[:] = 0
        for q in self.sq: q[:] = 0

    def process(self, x):
        self.inbuf[:-M] = self.inbuf[M:]; self.inbuf[-M:] = x
        S = np.fft.rfft(self.inbuf * self.wa)
        m, self.h = self.s.run(None, {"mag": np.abs(S).astype(np.float32)[None, None], "h": self.h})
        m = np.maximum(m[0, 0], self.floor)
        if self.la:                             # lookahead: this mask belongs to the frame LOOKAHEAD hops back
            self.sq.append(S); S = self.sq.pop(0)
        seg = np.fft.irfft(S * m, K)[K - 2 * M:].astype(np.float32) * self.ws
        out = self.tail + seg[:M]; self.tail = seg[M:].copy()
        return out


def process_file(x, **kw):
    """Whole signal through the streaming engine, re-aligned to the input (the engine runs (10 + 5*LOOKAHEAD) ms late)."""
    e = StreamEngine(**kw); x = np.asarray(x, np.float32); n = len(x); d = (1 + LOOKAHEAD) * M
    xp = np.concatenate([x, np.zeros(d + M, np.float32)]); xp = xp[: (len(xp) // M) * M]
    y = np.concatenate([e.process(xp[i:i + M]) for i in range(0, len(xp), M)])
    return y[d: d + n]
