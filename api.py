#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FramePack API Server
====================

FastAPI service that exposes a /generate endpoint for image-to-video.
This version mirrors demo_gradio.py's checkpoint/cache behavior:

- Uses HF_HOME=<repo>/hf_download (same as the Gradio app)
- Derives HF_HUB_CACHE from HF_HOME (and sets TRANSFORMERS_CACHE/DIFFUSERS_CACHE for safety)
- Also passes cache_dir=... explicitly to all from_pretrained() calls so the API
  ALWAYS reuses the same local cache (no redownloads when WebUI already fetched files).

CLI:
  python api.py --api --port 7000 [--unload]

Flags:
  --api         Launch FastAPI server
  --port <int>  Port (default 7000)
  --unload      Unload models from GPU/CPU after each request (frees VRAM/RAM)

This endpoint accepts: multipart/form-data { image: <file>, prompt: <str>, duration?: <float>, steps?: <int>, crf?: <int>, unload?: <bool> }
and returns: MP4 bytes (video/mp4)
"""

from __future__ import annotations

import argparse
import io
import os
import math
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import einops
import safetensors.torch as sf
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── FramePack / Diffusers helper imports (from this repository) ───────────────
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
    vae_decode_fake,
)
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    generate_timestamp,
)
from diffusers_helper.models.hunyuan_video_packed import (
    HunyuanVideoTransformer3DModelPacked,
)
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu,
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete,
)
from diffusers_helper.thread_utils import AsyncStream, async_run

# ──────────────────────────────────────────────────────────────────────────────
# Cache / directories: Make API behave like demo_gradio.py
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
HF_HOME = BASE_DIR / "hf_download"
HF_HOME.mkdir(parents=True, exist_ok=True)

# Mirror demo_gradio.py: store everything under ./hf_download
os.environ["HF_HOME"] = str(HF_HOME)
# Explicit caches (safe defaults). Hugging Face recommends HF_HOME/HF_HUB_CACHE;
# TRANSFORMERS_CACHE/DIFFUSERS_CACHE are also honored by downstream libs.
os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))           # repos cache:contentReference[oaicite:7]{index=7}
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))     # safe co-location
os.environ.setdefault("DIFFUSERS_CACHE", str(HF_HOME / "hub"))        # safe co-location

# Also use explicit cache_dir in all from_pretrained() for belt & suspenders
CACHE_DIR = os.environ.get("HF_HUB_CACHE", str(HF_HOME / "hub"))

# Output directory like the Gradio app
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Model globals (loaded once)
# ──────────────────────────────────────────────────────────────────────────────

text_encoder: Optional[LlamaModel] = None
text_encoder_2: Optional[CLIPTextModel] = None
tokenizer: Optional[LlamaTokenizerFast] = None
tokenizer_2: Optional[CLIPTokenizer] = None
vae: Optional[AutoencoderKLHunyuanVideo] = None
feature_extractor: Optional[SiglipImageProcessor] = None
image_encoder: Optional[SiglipVisionModel] = None
transformer: Optional[HunyuanVideoTransformer3DModelPacked] = None

high_vram: bool = False
stream = AsyncStream()  # basic async worker bridge (used in demo; harmless here)

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def pil_from_upload(upload: UploadFile) -> Image.Image:
    try:
        data = upload.file.read() if hasattr(upload, "file") else upload.read()
        if hasattr(upload, "seek"):
            upload.seek(0)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        return im
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Model loading mirrors demo_gradio.py exactly (but with cache_dir)
# ──────────────────────────────────────────────────────────────────────────────

def load_models_once():
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, feature_extractor, image_encoder, transformer
    global high_vram

    if all([
        text_encoder is not None,
        text_encoder_2 is not None,
        tokenizer is not None,
        tokenizer_2 is not None,
        vae is not None,
        feature_extractor is not None,
        image_encoder is not None,
        transformer is not None,
    ]):
        return  # already loaded

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60
    print(f"[FramePack API] Free VRAM: {free_mem_gb:.2f} GB  |  High-VRAM mode: {high_vram}")

    # -- Repos as in demo_gradio.py
    HY_REPO = "hunyuanvideo-community/HunyuanVideo"
    FLUX_REPO = "lllyasviel/flux_redux_bfl"
    TRANS_REPO = "lllyasviel/FramePackI2V_HY"

    # -- Load on CPU initially (like demo), set dtypes, then move smartly
    text_encoder = LlamaModel.from_pretrained(
        HY_REPO, subfolder="text_encoder", torch_dtype=torch.float16, cache_dir=CACHE_DIR
    ).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(
        HY_REPO, subfolder="text_encoder_2", torch_dtype=torch.float16, cache_dir=CACHE_DIR
    ).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained(HY_REPO, subfolder="tokenizer", cache_dir=CACHE_DIR)
    tokenizer_2 = CLIPTokenizer.from_pretrained(HY_REPO, subfolder="tokenizer_2", cache_dir=CACHE_DIR)
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        HY_REPO, subfolder="vae", torch_dtype=torch.float16, cache_dir=CACHE_DIR
    ).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained(
        FLUX_REPO, subfolder="feature_extractor", cache_dir=CACHE_DIR
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        FLUX_REPO, subfolder="image_encoder", torch_dtype=torch.float16, cache_dir=CACHE_DIR
    ).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        TRANS_REPO, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR
    ).cpu()

    # Eval & dtype setup per demo
    for m in (vae, text_encoder, text_encoder_2, image_encoder, transformer):
        m.eval()

    transformer.high_quality_fp32_output_for_inference = True
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()
        # Install faster offload
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)

# ──────────────────────────────────────────────────────────────────────────────
# Core generation (adapted from demo_gradio.py -> worker())
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_from_image(
    pil_image: Image.Image,
    prompt: str,
    *,
    total_second_length: float = 5.0,
    steps: int = 25,
    cfg: float = 1.0,
    gs: float = 10.0,
    rs: float = 0.0,
    gpu_memory_preservation: float = 6.0,
    latent_window_size: int = 9,
    mp4_crf: int = 16,
    unload_after: bool = False,
) -> Path:
    """
    Generates a short clip and returns the output filepath.
    """
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2
    global vae, feature_extractor, image_encoder, transformer
    assert transformer is not None and vae is not None

    # Prepare numpy image like demo
    np_img = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    H, W, C = np_img.shape
    height, width = resize_and_center_crop.__defaults__[1], resize_and_center_crop.__defaults__[0]  # not used: ensure flake
    height, width = (640, 640)  # will be resolved below

    # find_nearest_bucket from helpers
    from diffusers_helper.bucket_tools import find_nearest_bucket
    height, width = find_nearest_bucket(H, W, resolution=640)
    input_image_np = resize_and_center_crop(np_img, target_width=width, target_height=height)

    job_id = generate_timestamp()
    Image.fromarray(input_image_np).save(OUTPUTS_DIR / f"{job_id}.png")

    input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
    input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

    # Encode text
    if not high_vram:
        fake_diffusers_current_device(text_encoder, gpu)
        load_model_as_complete(text_encoder_2, target_device=gpu)

    llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
    if cfg == 1:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds("", text_encoder, text_encoder_2, tokenizer, tokenizer_2)

    llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

    # VAE encode
    if not high_vram:
        load_model_as_complete(vae, target_device=gpu)
    start_latent = vae_encode(input_image_pt, vae)

    # CLIP Vision
    if not high_vram:
        load_model_as_complete(image_encoder, target_device=gpu)
    image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
    image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

    # Dtypes align with transformer
    llama_vec = llama_vec.to(transformer.dtype)
    llama_vec_n = llama_vec_n.to(transformer.dtype)
    clip_l_pooler = clip_l_pooler.to(transformer.dtype)
    clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
    image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

    # Steps & windows
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    num_frames = latent_window_size * 4 - 3

    rnd = torch.Generator("cpu").manual_seed(31337)

    history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
    history_pixels = None
    total_generated_latent_frames = 0

    latent_paddings = reversed(range(total_latent_sections))
    if total_latent_sections > 4:
        latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

    # Main sampling loop (condensed from demo)
    for latent_padding in latent_paddings:
        is_last_section = latent_padding == 0
        latent_padding_size = latent_padding * latent_window_size

        indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
        clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
        clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

        clean_latents_pre = start_latent.to(history_latents)
        clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

        if not high_vram:
            unload_complete_models()
            move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        transformer.initialize_teacache(enable_teacache=True, num_steps=steps)

        def _callback(d):
            # Light preview – keep as in demo (no streaming over API)
            return

        generated_latents = sample_hunyuan(
            transformer=transformer,
            sampler="unipc",
            width=width,
            height=height,
            frames=num_frames,
            real_guidance_scale=cfg,
            distilled_guidance_scale=gs,
            guidance_rescale=rs,
            num_inference_steps=steps,
            generator=rnd,
            prompt_embeds=llama_vec,
            prompt_embeds_mask=llama_attention_mask,
            prompt_poolers=clip_l_pooler,
            negative_prompt_embeds=llama_vec_n,
            negative_prompt_embeds_mask=llama_attention_mask_n,
            negative_prompt_poolers=clip_l_pooler_n,
            device=gpu,
            dtype=torch.bfloat16,
            image_embeddings=image_encoder_last_hidden_state,
            latent_indices=latent_indices,
            clean_latents=clean_latents,
            clean_latent_indices=clean_latent_indices,
            clean_latents_2x=clean_latents_2x,
            clean_latent_2x_indices=clean_latent_2x_indices,
            clean_latents_4x=clean_latents_4x,
            clean_latent_4x_indices=clean_latent_4x_indices,
            callback=_callback,
        )

        if is_last_section:
            generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

        total_generated_latent_frames += int(generated_latents.shape[2])
        history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

        if not high_vram:
            offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
            load_model_as_complete(vae, target_device=gpu)

        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

        section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
        overlapped_frames = latent_window_size * 4 - 3

        current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
        history_pixels = current_pixels if history_pixels is None else soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

        if not high_vram:
            unload_complete_models()

        # write progressive (keep last)
        out_path = OUTPUTS_DIR / f"{job_id}_{total_generated_latent_frames}.mp4"
        save_bcthw_as_mp4(history_pixels, str(out_path), fps=30, crf=mp4_crf)

        if is_last_section:
            final_path = OUTPUTS_DIR / f"{job_id}_final.mp4"
            out_path.rename(final_path)
            if unload_after:
                # aggressively free memory after request
                try:
                    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception:
                    pass
            return final_path

    raise RuntimeError("Generation loop terminated unexpectedly.")

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────────────

def create_app(unload_default: bool = False) -> FastAPI:
    app = FastAPI(title="FramePack API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/generate", response_class=StreamingResponse)
    async def generate_endpoint(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        duration: Optional[float] = Form(None),
        steps: Optional[int] = Form(None),
        crf: Optional[int] = Form(None),
        unload: Optional[bool] = Form(None),
    ):
        try:
            load_models_once()

            pil_image = pil_from_upload(image)
            dur = float(duration) if duration is not None else 5.0
            stp = int(steps) if steps is not None else 25
            crf_v = int(crf) if crf is not None else 16
            do_unload = bool(unload) if unload is not None else unload_default

            out_path = generate_from_image(
                pil_image, prompt,
                total_second_length=dur,
                steps=stp,
                mp4_crf=crf_v,
                unload_after=do_unload,
            )

            def _iterfile():
                with open(out_path, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 512)
                        if not chunk:
                            break
                        yield chunk

            headers = {"Content-Disposition": f'attachment; filename="{out_path.name}"'}
            return StreamingResponse(_iterfile(), media_type="video/mp4", headers=headers)

        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health():
        return {"ok": True}

    return app

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")
    parser.add_argument("--port", type=int, default=7000)
    parser.add_argument("--unload", action="store_true", help="Unload models after each request")
    args = parser.parse_args()

    if args.api:
        load_models_once()
        import uvicorn

        uvicorn.run(create_app(unload_default=args.unload), host="0.0.0.0", port=args.port)
    else:
        print("Nothing to do. Use --api to start the HTTP server.")
