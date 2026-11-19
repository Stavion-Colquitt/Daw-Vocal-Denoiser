# Google ViSQOL v3 (Apache 2.0), built from source, for eval.py's music-quality score (audio mode, 48 kHz).
# Runs on any OS with Docker, so the Windows/macOS/Linux eval behaves the same.
#   docker build -t visqol:local tools/visqol
#   docker run --rm -v "<dir with wavs + pairs.csv>:/work" visqol:local --batch_input_csv /work/pairs.csv --results_csv /work/out.csv
FROM ubuntu:22.04 AS build
# pinned: the ViSQOL commit the published eval results were produced with
ARG VISQOL_COMMIT=38d0b0163e441047d4429bf07ad09e5b9031d02c
ENV DEBIAN_FRONTEND=noninteractive USE_BAZEL_VERSION=5.1.0
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential python3 python3-numpy \
        curl ca-certificates unzip zip && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel
RUN git clone https://github.com/google/visqol /visqol && cd /visqol && git checkout "$VISQOL_COMMIT" \
    && git rev-parse HEAD > /visqol/COMMIT
WORKDIR /visqol
RUN bazel build :visqol -c opt && cp bazel-bin/visqol /usr/local/bin/visqol-bin

FROM ubuntu:22.04
COPY --from=build /usr/local/bin/visqol-bin /usr/local/bin/visqol
COPY --from=build /visqol/model /visqol/model
COPY --from=build /visqol/COMMIT /visqol/COMMIT
WORKDIR /visqol
ENTRYPOINT ["visqol", "--similarity_to_quality_model", "/visqol/model/libsvm_nu_svr_model.txt"]
