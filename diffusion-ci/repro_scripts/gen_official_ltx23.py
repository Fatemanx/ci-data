import argparse
import gc
import json
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import imageio
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

import ltx_pipelines.utils.denoisers as ltx_denoisers
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    LTX_2_3_HQ_PARAMS,
    detect_params,
)
from ltx_pipelines.utils.helpers import modality_from_latent_state

OUT_DIR = Path("/tmp/mmgen-official-ltx-report")
ASSET_ROOT = Path("/tmp/mmgen-official-ltx-assets")
IMAGE_URL = (
    "https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/"
    "5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg"
)
SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT = False

T2V_PROMPT = "A curious raccoon"
TI2V_PROMPT = (
    "The man in the picture slowly turns his head, his expression enigmatic "
    "and otherworldly. The camera performs a slow, cinematic dolly out, "
    "focusing on his face. Moody lighting, neon signs glowing in the "
    "background, shallow depth of field."
)


@dataclass(frozen=True)
class LTXCaseSpec:
    repo_id: str
    checkpoint_name: str
    distilled_lora_name: str | None
    upsampler_name: str | None
    pipeline: str
    num_gpus: int
    prompt: str
    height: int = 512
    width: int = 768
    num_frames: int = 25
    fps: int = 24
    seed: int = 42
    images: bool = False


CASE_SPECS: dict[str, LTXCaseSpec] = {
    "ltx_2_two_stage_t2v": LTXCaseSpec(
        repo_id="Lightricks/LTX-2",
        checkpoint_name="ltx-2-19b-dev.safetensors",
        distilled_lora_name="ltx-2-19b-distilled-lora-384.safetensors",
        upsampler_name="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        pipeline="two_stage",
        num_gpus=2,
        prompt=T2V_PROMPT,
        num_frames=24,
        seed=10,
    ),
    "ltx_2.3_two_stage_t2v_2gpus": LTXCaseSpec(
        repo_id="Lightricks/LTX-2.3",
        checkpoint_name="ltx-2.3-22b-dev.safetensors",
        distilled_lora_name="ltx-2.3-22b-distilled-lora-384.safetensors",
        upsampler_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        pipeline="two_stage",
        num_gpus=2,
        prompt=T2V_PROMPT,
    ),
    "ltx_2_3_two_stage_ti2v_2gpus": LTXCaseSpec(
        repo_id="Lightricks/LTX-2.3",
        checkpoint_name="ltx-2.3-22b-dev.safetensors",
        distilled_lora_name="ltx-2.3-22b-distilled-lora-384.safetensors",
        upsampler_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        pipeline="two_stage",
        num_gpus=2,
        prompt=TI2V_PROMPT,
        images=True,
    ),
    "ltx_2.3_one_stage_ti2v": LTXCaseSpec(
        repo_id="Lightricks/LTX-2.3",
        checkpoint_name="ltx-2.3-22b-dev.safetensors",
        distilled_lora_name=None,
        upsampler_name=None,
        pipeline="one_stage",
        num_gpus=2,
        prompt=TI2V_PROMPT,
        images=True,
    ),
    "ltx_2_3_hq_pipeline": LTXCaseSpec(
        repo_id="Lightricks/LTX-2.3",
        checkpoint_name="ltx-2.3-22b-dev.safetensors",
        distilled_lora_name="ltx-2.3-22b-distilled-lora-384.safetensors",
        upsampler_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        pipeline="two_stage_hq",
        num_gpus=1,
        prompt="Doraemon is eating dorayaki",
        height=1024,
        width=1024,
        num_frames=24,
    ),
}


def _consistency_gt_filenames(case_id: str, num_gpus: int, is_video: bool) -> list[str]:
    if is_video:
        return [
            f"{case_id}_{num_gpus}gpu_frame_0.png",
            f"{case_id}_{num_gpus}gpu_frame_mid.png",
            f"{case_id}_{num_gpus}gpu_frame_last.png",
        ]
    return [f"{case_id}_{num_gpus}gpu.jpg"]


def extract_key_frames_from_video(video_bytes: bytes) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise ValueError("Video has no frames")

        frames = []
        for idx in (0, total_frames // 2, total_frames - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Failed to read frame at index {idx}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames
    finally:
        os.unlink(tmp_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--case-ids", nargs="+", default=sorted(CASE_SPECS))
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--quantization",
        choices=["none", "fp8-cast", "fp8-scaled-mm"],
        default="fp8-cast",
    )
    parser.add_argument("--decode-audio", action="store_true")
    parser.add_argument(
        "--skip-v2a-cross-attn-for-video-gt",
        action="store_true",
        help="Disable the video-to-audio cross-attention branch to reproduce legacy CI GT.",
    )
    return parser.parse_args()


def resolve_quantization(name: str):
    if name == "none":
        return None
    if name == "fp8-cast":
        return QuantizationPolicy.fp8_cast()
    return QuantizationPolicy.fp8_scaled_mm()


class NoopAudioDecoder:
    def __call__(self, latent):
        return None


def sequential_guided_denoise(
    transformer,
    video_state,
    audio_state,
    sigma,
    video_guider,
    audio_guider,
    v_context,
    a_context,
    *,
    last_denoised_video,
    last_denoised_audio,
    step_index: int,
):
    v_skip = video_guider.should_skip_step(step_index)
    a_skip = audio_guider.should_skip_step(step_index)

    if v_skip and a_skip:
        return last_denoised_video, last_denoised_audio

    def maybe_skip_v2a(perturbation: PerturbationConfig) -> PerturbationConfig:
        if not SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT:
            return perturbation
        perturbations = list(perturbation.perturbations or [])
        perturbations.append(
            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None)
        )
        return PerturbationConfig(perturbations)

    passes = [("cond", v_context, a_context, PerturbationConfig.empty())]
    if (
        video_guider.do_unconditional_generation()
        or audio_guider.do_unconditional_generation()
    ):
        v_neg = (
            video_guider.negative_context
            if video_guider.negative_context is not None
            else v_context
        )
        a_neg = (
            audio_guider.negative_context
            if audio_guider.negative_context is not None
            else a_context
        )
        passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

    stg_perturbations = []
    if video_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                blocks=video_guider.params.stg_blocks,
            )
        )
    if audio_guider.do_perturbed_generation():
        stg_perturbations.append(
            Perturbation(
                type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                blocks=audio_guider.params.stg_blocks,
            )
        )
    if stg_perturbations:
        passes.append(
            ("ptb", v_context, a_context, PerturbationConfig(stg_perturbations))
        )

    if (
        video_guider.do_isolated_modality_generation()
        or audio_guider.do_isolated_modality_generation()
    ):
        passes.append(
            (
                "mod",
                v_context,
                a_context,
                PerturbationConfig(
                    [
                        Perturbation(
                            type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None
                        ),
                        Perturbation(
                            type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None
                        ),
                    ]
                ),
            )
        )

    results = {}
    for name, video_context, audio_context, perturbation in passes:
        video = (
            modality_from_latent_state(
                video_state, video_context, sigma, enabled=not v_skip
            )
            if video_state is not None
            else None
        )
        audio = (
            modality_from_latent_state(
                audio_state, audio_context, sigma, enabled=not a_skip
            )
            if audio_state is not None
            else None
        )
        results[name] = transformer(
            video=video,
            audio=audio,
            perturbations=BatchedPerturbationConfig([maybe_skip_v2a(perturbation)]),
        )

    cond_v, cond_a = results["cond"]
    uncond_v, uncond_a = results.get("uncond", (0.0, 0.0))
    ptb_v, ptb_a = results.get("ptb", (0.0, 0.0))
    mod_v, mod_a = results.get("mod", (0.0, 0.0))

    denoised_video = (
        last_denoised_video
        if v_skip
        else video_guider.calculate(cond_v, uncond_v, ptb_v, mod_v)
    )
    denoised_audio = (
        last_denoised_audio
        if a_skip
        else audio_guider.calculate(cond_a, uncond_a, ptb_a, mod_a)
    )
    return denoised_video, denoised_audio


def enable_low_memory_official_ltx() -> None:
    if SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT:
        ltx_denoisers._guided_denoise = sequential_guided_denoise


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        os.link(src, dst)


def prepare_gemma_root(spec: LTXCaseSpec) -> Path:
    safe_name = spec.repo_id.replace("/", "__")
    root = ASSET_ROOT / safe_name
    gemma_root = root / "gemma_root"
    gemma_root.mkdir(parents=True, exist_ok=True)

    if spec.repo_id == "Lightricks/LTX-2.3":
        source_root = Path(
            snapshot_download(
                repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                ignore_patterns=["*.onnx", "*.msgpack"],
                max_workers=8,
            )
        )
        for src in source_root.iterdir():
            if src.is_file():
                link(src, gemma_root / src.name)
        return gemma_root

    source_root = Path(
        snapshot_download(
            repo_id=spec.repo_id,
            allow_patterns=["text_encoder/*", "tokenizer/*"],
            max_workers=8,
        )
    )
    for src_dir in (source_root / "text_encoder", source_root / "tokenizer"):
        for src in src_dir.iterdir():
            if src.is_file():
                link(src, gemma_root / src.name)
    return gemma_root


def prepare_lora(
    path: str | None, strength: float = 1.0
) -> list[LoraPathStrengthAndSDOps]:
    if path is None:
        return []
    return [
        LoraPathStrengthAndSDOps(
            path,
            strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
    ]


def collect_video_frames(video_iter) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for chunk in video_iter:
        arr = chunk.detach().to("cpu").numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.extend([frame[..., :3] for frame in arr])
    return frames


def ci_key_frames(frames: list[np.ndarray], fps: int) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        imageio.mimsave(
            tmp_path,
            frames,
            fps=fps,
            format="mp4",
            codec="libx264",
            quality=5,
        )
        return extract_key_frames_from_video(Path(tmp_path).read_bytes())
    finally:
        os.unlink(tmp_path)


def save_case(
    case_id: str, frames: list[np.ndarray], fps: int, num_gpus: int
) -> list[str]:
    selected = ci_key_frames(frames, fps)
    saved: list[str] = []
    for frame, filename in zip(
        selected,
        _consistency_gt_filenames(case_id, num_gpus, is_video=True),
        strict=True,
    ):
        Image.fromarray(frame).save(OUT_DIR / filename)
        saved.append(filename)
    return saved


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_memory_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "memory_allocated": int(torch.cuda.memory_allocated()),
        "memory_reserved": int(torch.cuda.memory_reserved()),
        "max_memory_allocated": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved": int(torch.cuda.max_memory_reserved()),
    }


def image_conditioning(
    input_image_path: str, enabled: bool
) -> list[ImageConditioningInput]:
    if not enabled:
        return []
    return [
        ImageConditioningInput(
            path=input_image_path,
            frame_idx=0,
            strength=1.0,
            crf=33,
        )
    ]


def run_case(
    case_id: str,
    spec: LTXCaseSpec,
    args: argparse.Namespace,
    device: torch.device,
    quantization: QuantizationPolicy | None,
    input_image_path: str,
) -> dict:
    checkpoint_name = args.checkpoint_name or spec.checkpoint_name
    checkpoint_path = hf_hub_download(spec.repo_id, checkpoint_name)
    params = (
        LTX_2_3_HQ_PARAMS
        if spec.pipeline == "two_stage_hq"
        else detect_params(checkpoint_path)
    )
    gemma_root = str(prepare_gemma_root(spec))
    distilled_lora_path = (
        hf_hub_download(spec.repo_id, spec.distilled_lora_name)
        if spec.distilled_lora_name
        else None
    )
    upsampler_path = (
        hf_hub_download(spec.repo_id, spec.upsampler_name)
        if spec.upsampler_name
        else None
    )

    height = args.height or spec.height
    width = args.width or spec.width
    num_frames = args.num_frames or spec.num_frames
    fps = args.fps or spec.fps
    steps = args.steps or params.num_inference_steps
    images = image_conditioning(input_image_path, spec.images)

    print(f"[ltx-official] generating {case_id}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if spec.pipeline == "one_stage":
        pipe = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            loras=[],
            device=device,
            quantization=quantization,
        )
    elif spec.pipeline == "two_stage":
        pipe = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=prepare_lora(distilled_lora_path),
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            device=device,
            quantization=quantization,
        )
    elif spec.pipeline == "two_stage_hq":
        pipe = TI2VidTwoStagesHQPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=prepare_lora(distilled_lora_path),
            distilled_lora_strength_stage_1=0.25,
            distilled_lora_strength_stage_2=0.5,
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_root,
            loras=(),
            device=device,
            quantization=quantization,
        )
    else:
        raise ValueError(f"unknown LTX pipeline kind: {spec.pipeline}")

    if not args.decode_audio:
        pipe.audio_decoder = NoopAudioDecoder()

    tiling_config = TilingConfig.default() if spec.pipeline != "one_stage" else None
    call_kwargs = dict(
        prompt=spec.prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        seed=spec.seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        video_guider_params=params.video_guider_params,
        audio_guider_params=params.audio_guider_params,
        images=images,
        max_batch_size=1,
    )
    if tiling_config is not None:
        call_kwargs["tiling_config"] = tiling_config

    video, audio = pipe(**call_kwargs)
    frames = collect_video_frames(video)
    saved = save_case(case_id, frames, fps=fps, num_gpus=spec.num_gpus)
    result = {
        "case_id": case_id,
        "repo_id": spec.repo_id,
        "pipeline_class": pipe.__class__.__name__,
        "saved_files": saved,
        "height": height,
        "width": width,
        "num_frames": len(frames),
        "fps": fps,
        "num_inference_steps": steps,
        "checkpoint_path": checkpoint_path,
        "distilled_lora_path": distilled_lora_path,
        "upsampler_path": upsampler_path,
        "image_path": input_image_path if spec.images else None,
        "cuda_memory": cuda_memory_snapshot(),
    }
    if tiling_config is not None:
        result["video_chunks_number"] = get_video_chunks_number(
            num_frames, tiling_config
        )

    del pipe, video, audio, frames
    cleanup_cuda()
    return result


def main() -> None:
    global OUT_DIR, SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT
    args = parse_args()
    OUT_DIR = Path(args.out_dir)
    SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT = args.skip_v2a_cross_attn_for_video_gt
    unknown_cases = sorted(set(args.case_ids) - set(CASE_SPECS))
    if unknown_cases:
        raise ValueError(f"Unknown LTX official case id(s): {' '.join(unknown_cases)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enable_low_memory_official_ltx()
    torch.backends.cuda.enable_cudnn_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    device = torch.device(args.device)
    quantization = resolve_quantization(args.quantization)
    input_image_path = str(Path(tempfile.gettempdir()) / "ltx_ti2v_input.jpg")
    if not Path(input_image_path).exists():
        urlretrieve(IMAGE_URL, input_image_path)

    manifest = {
        "generator": "official-ltx-pipelines",
        "args": vars(args),
        "low_memory_overrides": {
            "torch_inference_mode": True,
            "fp8_quantization": args.quantization,
            "component_lifecycle": "official component lifecycle; no layer/block streaming override",
            "guided_denoise": (
                "legacy sequential guidance with global V2A skip"
                if SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT
                else "official batched guidance passes (max_batch_size=1 default)"
            ),
            "skip_v2a_cross_attn_for_video_gt": SKIP_V2A_CROSS_ATTN_FOR_VIDEO_GT,
            "decode_audio": bool(args.decode_audio),
            "attention": "official AttentionFunction.DEFAULT",
        },
        "cases": [],
        "failures": [],
        "time": time.time(),
    }

    for case_id in args.case_ids:
        try:
            manifest["cases"].append(
                run_case(
                    case_id=case_id,
                    spec=CASE_SPECS[case_id],
                    args=args,
                    device=device,
                    quantization=quantization,
                    input_image_path=input_image_path,
                )
            )
        except Exception as exc:
            manifest["failures"].append(
                {
                    "case_id": case_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "cuda_memory": cuda_memory_snapshot(),
                }
            )
            cleanup_cuda()

    (OUT_DIR / "official_ltx_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
