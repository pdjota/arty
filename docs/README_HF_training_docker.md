---
title: Arty Train CNN-RNN
emoji: 🐨
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false
---

# HF training Space (Docker)

Use this README as the **Space README** only for a **dedicated training Space** (e.g. `pdjota/arty-train-cnnrnn`), not for the Gradio demo Space.

The repo [`Dockerfile`](../Dockerfile) runs dataset materialization, training, and model upload. See comments in that file for environment variables.

The **Gradio demo** Space should use **`sdk: gradio`** in the **root** [`README.md`](../README.md), not Docker.
