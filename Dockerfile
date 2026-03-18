# Dockerfile for a dedicated HF training Space (GPU)
# It downloads the dataset, trains CNN-RNN, and uploads the checkpoint.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Dataset + training + upload configuration (override via Space Variables)
ENV HF_DATASET_ID=pdjota/artyset
ENV MODEL_REPO_ID=pdjota/arty-cnn-rnn
ENV ARCH=cnnrnn
ENV EPOCHS=5
ENV BATCH_SIZE_PRIMARY=32
ENV BATCH_SIZE_FALLBACK=16

# Space secret `HF_TOKEN` must be provided for uploading/downloading.
CMD bash -lc "\
  # Keep the container healthy while long-running training is happening. \
  python -m http.server 7860 --bind 0.0.0.0 >/dev/null 2>&1 & \
  SERVER_PID=$!; \
  trap 'kill ${SERVER_PID} >/dev/null 2>&1 || true' EXIT; \
  python scripts/materialize_artyset_from_hf.py --repo-id \"$HF_DATASET_ID\" --data-dir data && \
  python scripts/train_cnn_safe.py --arch \"$ARCH\" --epochs \"$EPOCHS\" --batch-size-primary \"$BATCH_SIZE_PRIMARY\" --batch-size-fallback \"$BATCH_SIZE_FALLBACK\" && \
  python scripts/upload_model_to_hf.py --repo-id \"$MODEL_REPO_ID\" --checkpoint \"checkpoints/$ARCH/best.pt\" --export-labels-dir data/label_maps \
"

