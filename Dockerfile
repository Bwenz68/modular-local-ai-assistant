# ── Dockerfile ─────────────────────────────────────────────
FROM mambaorg/micromamba:1.5.8-jammy-cuda-12.4.1

# 0) Run system installs as root
USER root

# 1) Voice system libs
#    ffmpeg: audio encode/decode for Whisper
#    libespeak-ng1: phoneme generation for Piper
#    portaudio19-dev: ALSA interface for sounddevice
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libespeak-ng1 \
      portaudio19-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) Re-create the binary environment exactly as on host
COPY micromamba_env.yaml /tmp/
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN micromamba env create -f /tmp/micromamba_env.yaml && \
    micromamba run -n aiassist pip install --no-cache-dir \
      faster-whisper==1.0.3 \
      sounddevice~=0.4.6 && \
    micromamba clean --all --yes

# 3) Voice‐model directory
ENV PIPER_VOICE_DIR=/data/piper_models

# 4) Switch back to bash login shell (keeps conda env active)
SHELL ["bash", "-l", "-c"]
ENV PATH /opt/conda/envs/aiassist/bin:$PATH

# 5) Copy project code
WORKDIR /app
COPY . /app

# 6) Default entrypoint
CMD ["python", "-m", "assistant.main"]
