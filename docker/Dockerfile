# DAW Vocal Denoiser: CPU-only container (ONNX Runtime, no PyTorch), any version.
# Put that version's daw_denoiser.onnx (from Hugging Face) in docker/, then from the repo root:
#   docker build -f docker/Dockerfile --build-arg VERSION=v1d --build-arg DAW_LOOKAHEAD=1 -t daw-denoiser:v1d .
#   docker build -f docker/Dockerfile --build-arg VERSION=v1b --build-arg DAW_HIDDEN=384 -t daw-denoiser:v1b .
#   docker run --rm -v "${PWD}:/data" daw-denoiser:v1d in.wav out.wav       (or: ... serve, port 8000)
FROM python:3.12-slim
ARG VERSION=v1c
ARG DAW_HIDDEN=256
ARG DAW_LOOKAHEAD=0
LABEL org.opencontainers.image.title="DAW Vocal Denoiser ${VERSION}" \
      org.opencontainers.image.description="Real-time 48 kHz vocal denoiser, CPU only" \
      org.opencontainers.image.source="https://github.com/Stavion-Colquitt/Daw-Vocal-Denoiser" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 \
    DAW_VERSION=${VERSION} DAW_HIDDEN=${DAW_HIDDEN} DAW_LOOKAHEAD=${DAW_LOOKAHEAD}
RUN pip install --no-cache-dir numpy==2.2.6 onnxruntime==1.22.1 soundfile==0.13.1 soxr==0.5.0.post1
WORKDIR /app
COPY denoiser/stream.py denoiser/denoise.py docker/daw_denoiser.onnx ./
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
USER app
WORKDIR /data
EXPOSE 8000
ENTRYPOINT ["python", "/app/denoise.py"]
CMD ["--help"]
