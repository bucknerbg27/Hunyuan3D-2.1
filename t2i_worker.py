"""
Text-to-image worker using SDXL 1.0 base.

Fits in ~7GB VRAM with model_cpu_offload. Designed to work alongside
the Hunyuan3D shape/texture pipelines within a 32GB budget.
"""
import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image


class Text2ImageWorker:
    """Text-to-image generation using SDXL 1.0 base with CPU offload."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pipe = None

    def load(self):
        """Load the pipeline with CPU offload to minimize VRAM."""
        if self.pipe is not None:
            return
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
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
        steps: int = 30,
        guidance_scale: float = 7.5,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            seed: Random seed for reproducibility.
            steps: Number of inference steps (20-50 recommended).
            guidance_scale: CFG scale (5.0-10.0 recommended).

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
