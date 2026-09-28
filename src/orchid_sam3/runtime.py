"""GPU/source guards and strict base/finetuned checkpoint loading."""
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys

from .config import check_source
from .data import SAM3_COMMIT


def require_runtime(repo):
    if platform.system() != "Linux":
        raise RuntimeError("Training runtime targets Linux/WSL2 with NCCL; use --prepare-only on Windows")
    repo = check_source(repo)
    import torch
    import sam3
    if not Path(sam3.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Imported sam3 is not the pinned --sam3-repo checkout")
    if not torch.cuda.is_available() or "RTX 5090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expose the intended RTX 5090 as CUDA device 0 (CUDA_VISIBLE_DEVICES)")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Target GPU/runtime does not support bfloat16")
    return {
        "python": sys.version, "os": platform.platform(), "sam3_commit": SAM3_COMMIT,
        "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch_cuda": torch.version.cuda,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], text=True),
    }


def build_model(checkpoint_path, bpe_path, device="cpu", eval_mode=False):
    import torch
    from sam3.model_builder import build_sam3_image_model
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise ValueError("An existing SAM 3 base or project checkpoint is required")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    if any(k.startswith("detector.") for k in weights):
        weights = {k[len("detector."):]: v for k, v in weights.items() if k.startswith("detector.")}
    model = build_sam3_image_model(bpe_path=bpe_path, device="cpu", eval_mode=eval_mode,
                                  load_from_HF=False, enable_segmentation=True, compile=False)
    model.load_state_dict(weights, strict=True)
    return model.to(device)
