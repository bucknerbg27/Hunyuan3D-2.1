"""
Text-to-image worker using Stable Diffusion 3 Medium.

Fits in ~5GB VRAM with model_cpu_offload. Designed to work alongside
the Hunyuan3D shape/texture pipelines within a 32GB budget.
Unloaded after each generation to free VRAM for 3D models.

Requires HF_TOKEN env var (SD3 Medium is a gated model).
"""
import os
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline


class Text2ImageWorker:
    """Text-to-image generation using SD3 Medium with CPU offload."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None
        self.token = os.environ.get("HF_TOKEN")

    def load(self):
        """Load the pipeline with CPU offload to minimize VRAM."""
        if self.pipe is not None:
            return
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required. "
                "Accept the license at https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers "
                "and set HF_TOKEN to your HuggingFace access token."
            )
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            token=self.token,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.set_progress_bar_config(disable=True)

    def unload(self):
        """Free GPU memory by moving pipeline to CPU and clearing cache."""
        if self.pipe is not None:
            self.pipe.to("cpu")
            del self.pipe
            self.pipe = None
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 28,
        guidance_scale: float = 7.0,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            seed: Random seed for reproducibility.
            steps: Number of inference steps (20-40 recommended for SD3).
            guidance_scale: CFG scale (5.0-7.0 recommended for SD3).

        Returns:
            PIL Image in RGBA mode.
        """
        if self.pipe is None:
            self.load()

        generator = torch.Generator(device=self.device).manual_seed(seed)
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=1024,
            height=1024,
        ).images[0]

        return image.convert("RGBA")
