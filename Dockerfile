# Dockerfile for a dedicated HF training Space (GPU)
# It downloads the dataset, trains CNN-RNN, and uploads the checkpoint.

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
# Avoid libgomp warnings/crashes from a weird OMP_NUM_THREADS value in some runtimes.
ENV OMP_NUM_THREADS=4

# Dataset + training + upload configuration (override via Space Variables)
ENV HF_DATASET_ID=pdjota/artyset
ENV MODEL_REPO_ID=pdjota/arty-cnn-rnn
ENV ARCH=cnnrnn
ENV EPOCHS=30
ENV BATCH_SIZE_PRIMARY=32
ENV BATCH_SIZE_FALLBACK=16
ENV EXTRA_EPOCHS=25

# Space secret `HF_TOKEN` must be provided for uploading/downloading.
CMD ["bash","-lc", "\
set -e; \
python -c 'import torch; print(\"[cuda] available=\", torch.cuda.is_available()); \
print(\"[cuda] name=\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")'; \
python -m http.server 7860 --bind 0.0.0.0 >/dev/null 2>&1 & SERVER_PID=$!; \
python scripts/materialize_artyset_from_hf.py --repo-id \"$HF_DATASET_ID\" --data-dir data --exclude-local-path \"Post_Impressionism/vincent-van-gogh_l-arlesienne-portrait-of-madame-ginoux-1890.jpg\"; \
if [ -f \"checkpoints/$ARCH/last.pt\" ]; then \
  echo \"[train] Resuming for ${EXTRA_EPOCHS} more epochs\"; \
  python scripts/train_cnn.py --arch \"$ARCH\" --resume --epochs \"$EXTRA_EPOCHS\" --batch-size \"$BATCH_SIZE_PRIMARY\" || \
  python scripts/train_cnn.py --arch \"$ARCH\" --resume --epochs \"$EXTRA_EPOCHS\" --batch-size \"$BATCH_SIZE_FALLBACK\"; \
else \
  echo \"[train] No last.pt found; training from scratch for ${EPOCHS} epochs\"; \
  python scripts/train_cnn_safe.py --arch \"$ARCH\" --epochs \"$EPOCHS\" --batch-size-primary \"$BATCH_SIZE_PRIMARY\" --batch-size-fallback \"$BATCH_SIZE_FALLBACK\"; \
fi; \
python scripts/upload_model_to_hf.py --repo-id \"$MODEL_REPO_ID\" --checkpoint \"checkpoints/$ARCH/best.pt\"; \
kill $SERVER_PID >/dev/null 2>&1 || true \
"]

