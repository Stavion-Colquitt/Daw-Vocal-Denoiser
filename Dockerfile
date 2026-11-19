# DAW Vocal Denoiser v1d: CPU-only container (ONNX Runtime, no PyTorch).
#   docker run --rm -v "${PWD}:/data" stavion/daw-denoiser in.wav out.wav
#   docker run --rm -p 8000:8000 stavion/daw-denoiser serve
FROM python:3.12-slim

LABEL org.opencontainers.image.title="DAW Vocal Denoiser v1d" \
      org.opencontainers.image.description="Real-time 48 kHz vocal denoiser (15 ms latency), CPU only" \
      org.opencontainers.image.source="https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser-v1d" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
RUN pip install --no-cache-dir numpy==2.2.6 onnxruntime==1.22.1 soundfile==0.13.1 soxr==0.5.0.post1

WORKDIR /app
COPY stream.py denoise.py daw_denoiser.onnx ./
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
USER app
WORKDIR /data
EXPOSE 8000
ENTRYPOINT ["python", "/app/denoise.py"]
CMD ["--help"]
