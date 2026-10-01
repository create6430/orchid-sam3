"""GPU/source guards and strict base/finetuned checkpoint loading."""
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
from types import MethodType

from .config import check_source
from .data import SAM3_COMMIT


def _trainable_mlp_forward(module, x):
    """Pinned upstream fused MLP is inference-only; preserve autograd in training."""
    import torch
    from sam3.model.vitdet import Mlp
    if not torch.is_grad_enabled():
        return Mlp.forward(module, x)
    x = module.act(module.fc1(x))
    x = module.drop1(x)
    x = module.norm(x)
    x = module.fc2(x)
    return module.drop2(x)


def _detach_semantic_output(module, inputs, output):
    # PLAN uses instance losses only. DDP must not expect gradients from this
    # auxiliary output; preserve its values and every checkpoint parameter.
    return output.detach()


def require_runtime(repo, smoke=False):
    if platform.system() != "Linux":
        raise RuntimeError("Training runtime targets Linux/WSL2 with NCCL; use --prepare-only on Windows")
    repo = check_source(repo)
    import torch
    import sam3
    if not Path(sam3.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Imported sam3 is not the pinned --sam3-repo checkout")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    allowed = ("NVIDIA GeForce RTX 5090", "NVIDIA GeForce RTX 3080") if smoke else ("NVIDIA GeForce RTX 5090",)
    if torch.cuda.get_device_name(0) not in allowed:
        raise RuntimeError(f"Expose one of {allowed} as CUDA device 0 (CUDA_VISIBLE_DEVICES)")
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
        # Official base also contains the SAM2 interactive neck. The image
        # builder disables that branch; all active image parameters stay strict.
        auxiliary = [k for k in weights if k.startswith("backbone.vision_backbone.sam2_convs.")]
        weights = {k: v for k, v in weights.items() if k not in auxiliary}
        print(f"Official base: excluded {len(auxiliary)} inactive SAM2 neck entries", flush=True)
    model = build_sam3_image_model(bpe_path=bpe_path, device="cpu", eval_mode=eval_mode,
                                  load_from_HF=False, enable_segmentation=True, compile=False)
    from sam3.model.vitdet import Mlp
    for module in model.modules():
        if isinstance(module, Mlp):
            module.forward = MethodType(_trainable_mlp_forward, module)
    model.segmentation_head.semantic_seg_head.register_forward_hook(_detach_semantic_output)
    model.load_state_dict(weights, strict=True)
    print(f"SAM3 strict checkpoint load succeeded: {checkpoint_path}; {len(weights)} entries", flush=True)
    return model.to(device)
