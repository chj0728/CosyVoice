# Accelerating CosyVoice with NVIDIA Triton Inference Server and TensorRT-LLM

Contributed by Yuekai Zhang (NVIDIA).

This repository provides three acceleration solutions for CosyVoice, each targeting a different model version and Token2Wav architecture. All solutions use TensorRT-LLM for LLM acceleration and NVIDIA Triton Inference Server for serving.

## Platform Compatibility

The Dockerfiles and `docker-compose.*.yml` examples in this directory are designed for NVIDIA's standard Triton TensorRT-LLM container flow on **Linux x86_64 + discrete NVIDIA GPU**.

For **Jetson Orin**, the current instructions are **not** a direct out-of-the-box deployment path:

- the provided Docker image and compose files are not Jetson-specific;
- Jetson deployment requires JetPack-compatible Triton/TensorRT-LLM builds or containers;
- some scripts may need additional adaptation because Jetson uses an ARM64 + integrated GPU environment.

In other words, **CosyVoice can only be deployed on Jetson Orin after manual Jetson-specific porting and validation**, not by following the commands in this directory unchanged.

## Solutions

### [CosyVoice3](README.Cosyvoice3.md)

Acceleration solution for [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512), the latest CosyVoice model. The pipeline includes `audio_tokenizer`, `speaker_embedding`, `token2wav`, and `vocoder` modules managed by Triton, with the LLM served via `trtllm-serve`.

### [CosyVoice2 + UNet Token2Wav](README.Cosyvoice2.Unet.md)

The baseline acceleration solution for CosyVoice2, using the original UNet-based flow-matching Token2Wav module.

### [CosyVoice2 + DiT Token2Wav](README.Cosyvoice2.DiT.md)

Replaces the UNet Token2Wav with a DiT-based Token2Wav module from [Step-Audio2](https://github.com/stepfun-ai/Step-Audio-2). Supports disaggregated deployment where the LLM and Token2Wav run on separate GPUs for better resource utilization under high concurrency.



## Quick Start

Each solution can be launched with a single Docker Compose command:

```sh
# CosyVoice3
docker compose -f docker-compose.cosyvoice3.yml up

# CosyVoice2 + UNet Token2Wav
docker compose -f docker-compose.cosyvoice2.unet.yml up

# CosyVoice2 + DiT Token2Wav
docker compose -f docker-compose.cosyvoice2.dit.yml up
```
