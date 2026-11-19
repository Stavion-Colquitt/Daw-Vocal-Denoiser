# DAW Vocal Denoiser

> **AI disclosure:** This project was built with AI assistance (Claude, by Anthropic) for writing code and
> documentation. All creative and technical decisions (model design, training data, evaluation) are mine, and I
> reviewed everything before publishing.

🎧 [Live A/B demo](https://huggingface.co/spaces/Stavion-Colquitt/Realtime_Denoisers) · 🤗 Models: [v1](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser), [v1c](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c), [v1b](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1b), [v1d](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1d) · 🐳 [`stavion/daw-denoiser`](https://hub.docker.com/r/stavion/daw-denoiser/tags) · 📋 [Project board](https://github.com/users/Stavion-Colquitt/projects/2)

A causal, real-time vocal denoiser for **tracking vocals in a DAW**: 48 kHz, **10 ms algorithmic latency** (15 ms with
lookahead), about 1.25M parameters, a fraction of a millisecond of CPU per 5 ms block. Trained from scratch on my own
vocal takes, VocalSet singing and DNS Challenge speech, against the noise a home studio actually has: room tone, hiss,
mains hum and headphone bleed.

Every version changes **one thing** from the one before it and is scored on the **same held-out clips**, so this repo
doubles as an ablation study: more data, a bigger model, and lookahead, measured against each other.

| Version | Question it asks | Latency | Params | CPU per 5 ms block | Folder |
|---|---|---|---|---|---|
| v1 | Baseline: does a small causal GRU meet the real-time rules? | 10 ms | 1.25M | 0.19 ms | [`versions/v1`](versions/v1) |
| v1c | Does more of my own singing help? | 10 ms | 1.25M | 0.24 ms | [`versions/v1c`](versions/v1c) |
| v1b | Bigger model (GRU 384) vs. more data? | 10 ms | 2.51M | 0.47 ms | [`versions/v1b`](versions/v1b) |
| v1d | What does 5 ms of lookahead buy? | 15 ms | 1.25M | 0.23 ms | [`versions/v1d`](versions/v1d) |

**Which one to use:** **v1d** for the best quality per CPU if 15 ms is fine for monitoring; **v1b** for the best
accuracy at 10 ms; **v1c** for my own voice with the cleanest pass-through; **v1** for other people's voices on a tight
CPU budget.

---

## Results (held-out test)

180 noisy clips (+ 20 clean pass-through clips) that no version trained on: my held-out song *Say Dat Name* (86 clips)
and held-out VocalSet singers (94 clips), mixed with dev-split noise at 0-40 dB SNR from a fixed seed. Every version runs
through the streaming engine on exactly the same clips, so differences are paired. Bold = best.

| Version | SI-SDR (dB) | SDR (dB) | PESQ-WB | ViSQOL | FAD-MERT (lower better) | Clean pass-through SI-SDR (dB) |
|---|---|---|---|---|---|---|
| Noisy input | 19.04 | 19.02 | 2.37 | 3.53 | 1.732 |  |
| v1 | 22.95 | 23.70 | 3.05 | 3.97 | **0.419** | 45.6 |
| v1c | 23.27 | 23.85 | 3.03 | 3.98 | 0.485 | **51.1** |
| v1b | **23.46** | **24.07** | 3.10 | **3.99** | 0.448 | 47.6 |
| v1d | 23.43 | 24.03 | **3.11** | **3.99** | 0.446 | 50.5 |

**Each one-variable step, paired** (mean ± 95% CI over clips; bold = the interval excludes zero):

| Step | SI-SDR, all clips | SI-SDR, my voice | SI-SDR, other singers | PESQ-WB, all clips |
|---|---|---|---|---|
| v1 → v1c (more own data) | +0.32 ± 0.36 | **+0.95 ± 0.67** | -0.27 ± 0.27 | **-0.02 ± 0.02** |
| v1c → v1b (GRU 256 → 384) | +0.19 ± 0.25 | -0.03 ± 0.45 | **+0.38 ± 0.24** | **+0.07 ± 0.02** |
| v1c → v1d (+5 ms lookahead) | +0.17 ± 0.19 | +0.09 ± 0.24 | +0.23 ± 0.30 | **+0.07 ± 0.02** |

**What this shows:** more of my own singing (v1c) specialized the model to my voice. A bigger model (v1b) is the most
accurate and wins back other singers. 5 ms of lookahead (v1d) buys about as much as doubling the model, at half the
CPU. The confidence intervals cover variation across test clips, not between training runs; a seed-variance re-run of
v1c is on the [project board](https://github.com/users/Stavion-Colquitt/projects/2). Raw per-clip scores: [`results/`](results).

---

## Quick start

```python
# pip install onnxruntime numpy      (download daw_denoiser.onnx for the version you want from Hugging Face)
import os
os.environ["DAW_ONNX"] = "path/to/daw_denoiser.onnx"
os.environ["DAW_LOOKAHEAD"] = "1"     # v1d only (0 for the others)
os.environ["DAW_HIDDEN"] = "256"      # 384 for v1b
from denoiser.stream import StreamEngine, process_file    # or run from inside denoiser/

eng = StreamEngine()
y_block = eng.process(x_block)        # float32[240] at 48 kHz in -> 240 denoised samples out
y = process_file(x)                   # offline: whole mono 48 kHz array, time-aligned output
```

| Version | `DAW_HIDDEN` | `DAW_LOOKAHEAD` | Weights |
|---|---|---|---|
| v1 | 256 | 0 | [https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser) |
| v1c | 256 | 0 | [https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1c) |
| v1b | 384 | 0 | [https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1b](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1b) |
| v1d | 256 | 1 | [https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1d](https://huggingface.co/Stavion-Colquitt/DAW_Vocal_Denoiser_v1d) |

## Docker

CPU-only images (ONNX Runtime, no PyTorch) on Docker Hub: `stavion/daw-denoiser:v1b`, `:v1d`, and `:latest` (v1c).
Any sample rate, mono or stereo; output is 24-bit WAV at the input's rate.

```bash
docker run --rm -v "${PWD}:/data" stavion/daw-denoiser:v1d vocal.wav vocal_clean.wav
docker run --rm -p 8000:8000 stavion/daw-denoiser:v1d serve
curl -X POST --data-binary @vocal.wav http://localhost:8000/denoise -o vocal_clean.wav
```

To build any version yourself, see [`docker/Dockerfile`](docker/Dockerfile). Container output matches `stream.py` run
directly (122.5 dB, limited by the 24-bit WAV output).

## Repository layout

| Path | What it is |
|---|---|
| `denoiser/model.py` | Model and asymmetric-window STFT (`DAW_HIDDEN`, `DAW_LOOKAHEAD`) |
| `denoiser/stream.py` | Real-time block engine and `process_file`; numpy + onnxruntime only |
| `denoiser/dataset.py`, `prep_data.py`, `prep_own_extra.py` | Training data: on-the-fly mixing, building the pools |
| `denoiser/train.py` | Training loop (compressed spectral + multi-resolution STFT loss, AdamW, EMA 0.999, MLflow) |
| `denoiser/export_onnx.py`, `check_parity.py` | One-frame ONNX export; streaming vs. PyTorch parity and CPU time |
| `denoiser/eval.py`, `compare.py`, `compare_multi.py` | Held-out evaluation; several versions on identical clips |
| `denoiser/test_lookahead.py` | Confirms lookahead reaches exactly as far ahead as set, and no further |
| `denoiser/denoise.py`, `docker/` | Container CLI and HTTP service |
| `versions/<v>/` | Notes, results and the training script (`run.ps1`) for each version |
| `results/` | Held-out comparison of all four versions, per-clip scores |
| `tools/visqol/` | ViSQOL v3 built from source in Docker (used by `eval.py`) |

Retrain a version: `versions/<v>/run.ps1`. It sets that version's settings and keeps its checkpoints, MLflow db and
ONNX export inside its own folder (`DAW_RUN_DIR`).

## Evaluation metrics

`eval.py` runs the held-out test through the streaming engine and reports two groups of metrics, noisy input vs
enhanced output:

| Metric | What it measures | Why it's here |
|--------|------------------|---------------|
| SI-SDR (dB) | Waveform closeness to the clean take, ignoring level | Standard enhancement metric; the main number here |
| PESQ-WB | ITU-T P.862.2 predicted speech quality (at 16 kHz) | Kept for comparison with speech denoisers; it was designed for speech, so treat it as a rough guide on singing |
| Clean pass-through SI-SDR | How untouched clean vocals come out | The "do no harm" check for tracking |
| **SDR (BSS-Eval, 512-tap)** | Filter-based SDR, tolerant of small EQ/phase changes | Recommended over SI-SDR for music in recent singing-voice work |
| **ViSQOL v3, audio mode** | Full-band (48 kHz) perceptual quality, MOS-LQO 1 to 4.73 | The usual intrusive quality metric for music; Google's code, built from source in Docker (`tools/visqol`) |
| **MERT-MSE** | Distance between [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) music embeddings of output and clean take | Embedding metrics track listening tests better than SDR on singing |
| **FAD-MERT** | Fréchet distance between all outputs and all clean takes in MERT embedding space | Distribution-level check that outputs "sound like" clean vocals |

Before ViSQOL and MERT, every signal is loudness-normalized to -23 LUFS (EBU R128) so level differences don't bias
the scores. Results are also broken down (mean improvement with 95% confidence intervals) by input SNR band, noise
type (DNS noise, headphone bleed, hiss, hum) and vocal source, and every item is written to `eval_results_items.csv`.

```bash
pip install -r requirements-eval.txt          # torch is assumed to be installed already
docker build -t visqol:local tools/visqol      # one time; eval.py also builds it automatically if missing
python eval.py 200                             # held-out test
python eval.py 40 --split dev                  # check the pipeline on dev vocals without touching the test set
python eval.py 200 --no-visqol --no-mert       # speech-style metrics + SDR only (no Docker, no model download)
```

The test mixtures come from a fixed seed, so every run scores exactly the same clips. Objective metrics are still
only a proxy: a listening test (MUSHRA, ITU-R BS.1534) on real noisy takes is the final word.

References: Le Roux et al. 2019 (SI-SDR); Vincent et al. 2006 and Scheibler 2022 (BSS-Eval, fast_bss_eval);
Chinen et al. 2020 (ViSQOL v3); Li et al. 2023 (MERT); Kilgour et al. 2019 (FAD); Gupta et al. 2015,
"On the Perceptual Relevance of Objective Source Separation Measures for Singing"; "Teaching Speech Enhancement
Models To Sing" ([arXiv 2607.11630](https://arxiv.org/abs/2607.11630)) on SDR, MERT-MSE and loudness matching for singing.

## Data paths

Every data location is an environment variable with a default inside the repo, so a fresh clone works without
editing code:

| Variable | Default | What goes there |
|----------|---------|-----------------|
| `DAW_DATA` | `data/daw-denoiser` (at the repo root) | Training pool written by `prep_data.py` (own_vocals/, vocalset/, bleed/) |
| `DNS_DATA` | `data/dns` | [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband data: `clean_fullband/` and `noise_fullband/` |
| `DAW_CORPUS` | `data/corpus` | Your own vocal takes, one folder per song (`<song>/*.wav`), read by `prep_data.py` |
| `DAW_MUSIC` | `data/ableton-music` | Ableton project folders scanned by `prep_own_extra.py` for extra vocal takes |

On Windows, set one for the current shell with `$env:DAW_DATA = "E:\my\data"` (PowerShell) or permanently with
`setx DAW_DATA "E:\my\data"`. On Linux/macOS: `export DAW_DATA=/my/data`. `data/` is gitignored.

## Training data and licenses

| Source | License |
|--------|---------|
| My own vocal takes and instrumentals | Mine (not distributed) |
| [VocalSet](https://zenodo.org/records/1442513) (Wilkins et al., ISMIR 2018) | CC BY 4.0 |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) fullband speech and noise (subsets) | Per source, see the DNS repo |

No commercial sample packs. Code is MIT licensed (see `LICENSE`).

## Limits

- For monitoring and cleaning vocal takes (singing or speech) recorded in untreated rooms.
- Removes noise, not reverb. Mono, 48 kHz only (resample other rates first).
- Not tested on heavy distortion or SNR below 0 dB; very loud headphone bleed can survive.
- v1d adds 5 ms of latency (15 ms total); still low enough to monitor through.

## History

v1c, v1b and v1d started as separate repositories. They now live here, with their full commit history; the old
repositories are archived and point to this one.
