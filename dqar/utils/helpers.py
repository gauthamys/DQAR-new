"""
Utility Helper Functions

Common utilities for loading models, saving images, and other tasks.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from typing import Optional, List, Union, Any
from PIL import Image


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_dit_model(
    model_name: str = "facebook/DiT-XL-2-256",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Load a DiT model from HuggingFace.

    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        dtype: Model data type

    Returns:
        Loaded DiT model
    """
    try:
        from diffusers import DiTPipeline

        pipeline = DiTPipeline.from_pretrained(model_name, torch_dtype=dtype)
        pipeline.to(device)

        # Extract just the transformer
        model = pipeline.transformer
        return model

    except ImportError:
        raise ImportError("Please install diffusers: pip install diffusers")
    except Exception as e:
        # Fallback: try loading with transformers
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
            model.to(device)
            return model
        except:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")


def create_scheduler(
    scheduler_type: str = "ddim",
    num_train_timesteps: int = 1000,
    beta_schedule: str = "linear",
    **kwargs,
) -> Any:
    """
    Create a diffusion noise scheduler.

    Args:
        scheduler_type: Type of scheduler ("ddim", "dpm_solver", "euler")
        num_train_timesteps: Number of training timesteps
        beta_schedule: Beta schedule type
        **kwargs: Additional scheduler arguments

    Returns:
        Configured scheduler
    """
    try:
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
        )

        scheduler_cls = {
            "ddim": DDIMScheduler,
            "dpm_solver": DPMSolverMultistepScheduler,
            "euler": EulerDiscreteScheduler,
        }.get(scheduler_type)

        if scheduler_cls is None:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return scheduler_cls(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            **kwargs,
        )

    except ImportError:
        raise ImportError("Please install diffusers: pip install diffusers")


def save_images(
    images: Union[torch.Tensor, List[Image.Image]],
    output_dir: str,
    prefix: str = "sample",
    format: str = "png",
):
    """
    Save images to disk.

    Args:
        images: Tensor of images (N, C, H, W) in [0, 1] or list of PIL Images
        output_dir: Output directory path
        prefix: Filename prefix
        format: Image format (png, jpg, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(images, torch.Tensor):
        # Convert tensor to PIL images
        images = tensor_to_pil(images)

    for idx, img in enumerate(images):
        filename = f"{prefix}_{idx:04d}.{format}"
        img.save(output_path / filename)


def tensor_to_pil(
    images: torch.Tensor,
    normalize: bool = True,
) -> List[Image.Image]:
    """
    Convert tensor to PIL images.

    Args:
        images: Tensor (N, C, H, W) or (C, H, W)
        normalize: Whether images are in [-1, 1] range

    Returns:
        List of PIL Images
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Move to CPU and convert to numpy
    images = images.cpu()

    if normalize:
        # Assume [-1, 1] range, convert to [0, 1]
        images = (images + 1) / 2

    # Clamp to valid range
    images = images.clamp(0, 1)

    # Convert to numpy (N, H, W, C) uint8
    images = (images * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    pil_images = []
    for img in images:
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            pil_images.append(Image.fromarray(img, mode="L"))
        else:
            pil_images.append(Image.fromarray(img))

    return pil_images


def pil_to_tensor(
    images: List[Image.Image],
    normalize: bool = True,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Convert PIL images to tensor.

    Args:
        images: List of PIL Images
        normalize: Whether to normalize to [-1, 1]
        device: Target device

    Returns:
        Tensor of shape (N, C, H, W)
    """
    tensors = []
    for img in images:
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to tensor
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        if normalize:
            tensor = tensor * 2 - 1  # [0, 1] -> [-1, 1]

        tensors.append(tensor)

    return torch.stack(tensors).to(device)


def load_prompts(
    source: Union[str, List[str]],
    max_prompts: Optional[int] = None,
) -> List[str]:
    """
    Load text prompts from file or list.

    Args:
        source: File path or list of prompts
        max_prompts: Maximum number of prompts to return

    Returns:
        List of prompts
    """
    if isinstance(source, list):
        prompts = source
    else:
        path = Path(source)
        if path.suffix == ".json":
            import json
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and "prompts" in data:
                    prompts = data["prompts"]
                else:
                    raise ValueError(f"Cannot parse prompts from {path}")
        else:
            # Assume plain text, one prompt per line
            with open(path) as f:
                prompts = [line.strip() for line in f if line.strip()]

    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    return prompts


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"
