# DAW Vocal Denoiser

> A causal, real-time vocal denoiser for **tracking vocals in a DAW**: 48 kHz, **10 ms algorithmic latency**,
> 1.25M parameters, 0.19 ms of CPU per 5 ms block. Trained from scratch on my own vocal takes, VocalSet singing
> and DNS Challenge speech, against the noise a home studio actually has: room tone, hiss, mains hum and
> headphone bleed.

Companion to [Audio_ML_Noise_Reduction](https://github.com/Stavion-Colquitt/Audio_ML_Noise_Reduction) (the
call/meeting denoiser). Same lab, different job: a call can afford 100+ ms; a singer hearing themselves through
headphones cannot.

---

## Results

Held-out singing the model never saw in training: a song of mine held out for test (`Say Dat Name`) plus held-out
VocalSet singers (female9, male11), mixed with dev-split noise (DNS noise, bleed, hiss, hum) at 0-40 dB SNR from a
fixed seed. 200 items, run through the **streaming engine** (exactly what a DAW would run); about 10% are clean-only
and give the pass-through number.

| Metric | Noisy input | Enhanced | Change |
|--------|-------------|----------|--------|
| SI-SDR (dB) | 19.04 | **22.95** | **+3.91** |
| PESQ (WB, at 16 kHz) | 2.37 | **3.05** | **+0.68** |
| Clean pass-through SI-SDR (dB) | | **45.6** | clean takes are left essentially untouched |

| Real-time property | Value |
|--------------------|-------|
| Algorithmic latency | **10 ms** (5 ms hop, 10 ms synthesis window) |
| CPU per 5 ms block (ONNX Runtime, desktop CPU) | **0.19 ms** (real-time factor 0.038) |
| Streaming vs offline parity | 111.9 dB (the block-by-block engine reproduces the training path) |
| Parameters | 1,250,563 |

The test mixes start at a fairly high SNR on purpose: tracking noise is usually quiet, and the hard requirement in a
DAW is "do no harm" to a clean take, which the 45.6 dB pass-through number checks.

---

## How it works

```
48 kHz mic, 240-sample (5 ms) blocks
   |
   v  Asymmetric windows (Wang et al. 2021, arXiv:2106.11794)
   |    analysis: 32 ms (1536 samples -> 769 bins, 31 Hz resolution: resolves low sung harmonics)
   |    synthesis: 10 ms (480 samples) -> latency is set by the SHORT window
   |    analysis x synthesis = Hann(480): perfect reconstruction when the mask is 1 (error 9.5e-7)
   v
|S|^0.3 -> Linear -> LayerNorm -> PReLU -> 2-layer GRU (256) -> Linear -> PReLU -> Linear -> sigmoid mask
   |       (the GRU carries context across frames, so 5 ms frames still "know" the last few seconds)
   v
mask x S (noisy phase kept) -> iFFT -> overlap-add -> 5 ms out
```

Keeping the noisy phase is deliberate: the call denoiser's v3 experiment showed a learned phase made quality worse.

## Training recipe

| | |
|---|---|
| Clean audio | own vocal takes 35% (individual takes exported from my sessions) / VocalSet 35% / DNS Challenge speech 30% |
| Noise | DNS noise, simulated headphone bleed (my own instrumentals, high-passed), white/pink hiss, 50/60 Hz hum |
| Augmentation | random EQ tilt (different mics/preamps), synthetic room on 30% (same room on input and target: it removes noise, not the room), SNR 0-40 dB, 10% clean-only |
| Segments | 4 s at 48 kHz (long enough for the GRU to learn context) |
| Loss | compressed spectral loss (Braun & Tashev, power 0.3: 0.7 magnitude + 0.3 complex) + 0.5 x multi-resolution STFT |
| Level | the model sees the real input level (as in a DAW); only the loss is level-normalised |
| Optimiser | AdamW 1e-3, 1k-step warmup, cosine decay, grad clip 3.0, EMA 0.999 (EMA weights are evaluated and exported) |
| Run | 108,773 steps (batch 32), self-sized to a 9-hour budget; best dev checkpoint at step 77,000 |
| Hardware | RTX 4060 (8 GB); resumed across a GPU driver reset and moved between GPUs via checkpoints |
| Tracking | MLflow (`view_mlflow.bat`) |

Data split is by song and by singer, fixed up front; the test song and test singers are never used for training or
model selection.

---

## Use it

**Get the model:** download `daw_denoiser.onnx` from the [Releases](../../releases) page and put it in this folder.

```python
from stream import StreamEngine
eng = StreamEngine()                  # loads daw_denoiser.onnx
y_block = eng.process(x_block)        # x_block: float32[240] at 48 kHz -> 240 denoised samples, 10 ms later
```

`stream.py` also has `process_file()` for offline use. A JUCE VST3/AU plugin wrapping the same ONNX model is the
next step.

## Train it

```bash
pip install torch numpy scipy soundfile onnx onnxruntime mlflow pesq pystoi
python prep_data.py all          # builds the 48 kHz pools (point it at your own takes; see the paths at the top)
python train.py --hours 9        # sizes the run to the time budget; checkpoints every 1000 steps
python train.py --resume         # continue after a crash or reboot
python export_onnx.py            # one-frame streaming ONNX (state in, state out)
python check_parity.py           # streaming engine vs training path + speed
python eval.py 200               # held-out singing: SI-SDR / PESQ / clean pass-through
```

On Windows, `start_training.bat` runs the whole thing in the background (train -> export -> parity -> eval);
`resume_training.bat` continues an interrupted run.

---

## Files

| File | What it is |
|------|------------|
| `model.py` | Asymmetric-window STFT, the GRU mask model, `enhance()` |
| `dataset.py` | On-the-fly mixtures (noise, bleed, hiss, hum, rooms, EQ) |
| `prep_data.py` | Builds the training pools from session takes, VocalSet and bleed sources |
| `train.py` | Training loop: loss, EMA, warmup + cosine, time-budget sizing, resume, MLflow |
| `stream.py` | Real-time block engine (ONNX Runtime) |
| `export_onnx.py` | Exports the one-frame streaming model |
| `check_parity.py` | Streaming vs offline parity and CPU cost per block |
| `eval.py` | Held-out evaluation |
| `*.bat`, `*.ps1` | Windows launchers (start, resume, resume pinned to a specific GPU, MLflow UI) |

## Data and credits

- **VocalSet** (Wilkins et al., ISMIR 2018), CC BY 4.0.
- **DNS Challenge** clean speech and noise (Microsoft), see the DNS Challenge repository for per-source licenses.
- **My own vocal takes and instrumentals** (used with my permission, obviously). No commercial sample packs are in
  the training data.
- Asymmetric windows: Wang, Z.-Q. et al., *"STFT-Domain Neural Speech Enhancement with Very Low Algorithmic
  Latency"*, arXiv:2106.11794. Compressed loss: Braun & Tashev, *"A consolidated view of loss functions for
  supervised deep learning-based speech enhancement"*, 2021.

## License

MIT (code). Model weights: same terms, with the dataset attributions above.
