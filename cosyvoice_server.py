import os
import sys
import argparse
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.utils.common import set_all_random_seed
from tqdm import tqdm

app = FastAPI()

import io
import wave


def generate_wav(model_output, sample_rate=24000):
    pcm_chunks = []

    for out in model_output:
        pcm = out["tts_speech"].detach().cpu().numpy()
        pcm = np.squeeze(pcm)
        pcm = (pcm * 32768).astype(np.int16)
        pcm_chunks.append(pcm)

    pcm_all = np.concatenate(pcm_chunks, axis=0)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_all.tobytes())

    buffer.seek(0)
    return buffer.read()


def generate_data(model_output):
    for out in model_output:
        tts_audio = (out["tts_speech"].cpu().numpy() * 32768).astype("int16")
        yield tts_audio.tobytes()


def generate_pcm_data(model_output):
    for out in model_output:
        pcm = out["tts_speech"].detach().cpu().numpy()
        pcm = np.squeeze(pcm)
        pcm = (pcm * 32768).astype(np.int16)
        yield pcm.tobytes()


# simplified presets-based endpoints
_PRESETS = {
    "default": (
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "./asset/zero_shot_prompt.wav",
    ),
    "zh": (
        "You are a helpful assistant.<|endofprompt|>转任福建路转运判官。",
        "./asset/zero_shot_zh.wav",
    ),
    "hard_zh": (
        "You are a helpful assistant.<|endofprompt|>在中国鸦片泛滥的年代，不同材质的烟枪甚至成为了身份和地位的象征。",
        "./asset/zero_shot_hard_zh.wav",
    ),
    "longshu_zh": (
        "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。",
        "./asset/sft_longshu_zh.wav",
    ),
    "longwan_zh": (
        "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。",
        "./asset/sft_longwan_zh.wav",
    ),
}


@app.get("/inference_zero_shot")
@app.post(
    "/inference_zero_shot",
    description="data_type: wav/pcm. preset: default/zh/hard_zh/longshu_zh/longwan_zh",
)
async def inference_zero_shot(
    tts_text: str = Form(), data_type: str = Form("pcm"), preset: str = Form("default")
):
    if preset not in _PRESETS:
        return Response(content=f"preset not found: {preset}", status_code=404)
    prompt_text, prompt_wav_path = _PRESETS[preset]
    print(f"Using preset: {preset}")
    print(f"Prompt text: {prompt_text}")
    print(f"Prompt wav: {prompt_wav_path}")

    # model_output = cosyvoice.inference_zero_shot(
    #     tts_text, prompt_text, prompt_wav_path, stream=True
    # )
    # if data_type == "wav":
    #     return Response(generate_wav(model_output), media_type="audio/wav")
    # return StreamingResponse(generate_pcm_data(model_output), media_type="audio/pcm")
    if args.load_vllm:
        for i in tqdm(range(100)):
            set_all_random_seed(i)
            model_output = cosyvoice.inference_zero_shot(
                tts_text, prompt_text, prompt_wav_path, stream=True
            )
            if data_type == "wav":
                return Response(generate_wav(model_output), media_type="audio/wav")
            return StreamingResponse(
                generate_pcm_data(model_output), media_type="audio/pcm"
            )
    else:
        model_output = cosyvoice.inference_zero_shot(
            tts_text, prompt_text, prompt_wav_path, stream=True
        )
        if data_type == "wav":
            return Response(generate_wav(model_output), media_type="audio/wav")
        return StreamingResponse(
            generate_pcm_data(model_output), media_type="audio/pcm"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # server port
    parser.add_argument("--port", type=int, default=50001)

    # model dir
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Fun-CosyVoice3-0.5B",
        help="local path or modelscope repo id",
    )

    # load_jit
    # parser.add_argument(
    #     "--load_jit",
    #     type=bool,
    #     default=False,
    #     help="whether to load JIT engine",
    # )

    # load_trt
    parser.add_argument(
        "--load_trt",
        type=bool,
        default=False,
        help="whether to load TensorRT engine",
    )

    # load_vllm
    parser.add_argument(
        "--load_vllm",
        type=bool,
        default=False,
        help="whether to load vLLM engine",
    )

    # fp16
    parser.add_argument(
        "--fp16",
        type=bool,
        default=False,
        help="whether to use fp16 precision",
    )

    args = parser.parse_args()
    try:
        cosyvoice = AutoModel(
            model_dir=args.model_dir,
            # load_jit=args.load_jit,
            load_trt=args.load_trt,
            load_vllm=args.load_vllm,
            fp16=args.fp16,
        )
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
