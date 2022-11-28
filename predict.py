import base64
import os
from io import BytesIO
from typing import Callable, Optional, List, Tuple

import requests
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    EulerDiscreteScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
)
from PIL import Image


MODEL_CACHE = "diffusers-cache"


def hijack_scheduler_step(
        scheduler,
        progress_callback: Optional[Callable],
        image_callback: Optional[Callable],
        callback_frequency: int = 0,
):
    assert callback_frequency >= 0
    previous_scheduler_step = scheduler.step
    def logging_step(*args, **kwargs):
        model_out = kwargs.get('model_out', args[0])
        timestep = kwargs.get('timestep', args[1] or 0)
        if callback_frequency == 0 or timestep % callback_frequency == 0:
            if image_callback:
                #image = self.vae.decode(1 / 0.18125 * latents)
                #image = (image / 2 + 0.5).clamp(0, 1)
                image = model_out[0, :, :, :]
                raise NotImplementedError("Debugging this stupid callback to hell.")
                #image_callback(numpy_to_pil(image.cpu().permute(0, 2, 3, 1).numpy()))
        previous_scheduler_step(*args, **kwargs)
    scheduler.step = logging_step


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print("Loading pipelines...")
        self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            revision="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to(self.device)

        self.inpaint_pipe = StableDiffusionInpaintPipelineLegacy(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=None,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to(self.device)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default=""),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        init_image: Path = Input(
            description="Initial image to generate variations of. Will be resized to the specified width and height",
            default=None,
        ),
        mask: Path = Input(
            description="Black and white image to use as mask for inpainting over init_image. Black pixels are inpainted and white pixels are preserved. Tends to work better with prompt strength of 0.5-0.7. Consider using https://replicate.com/andreasjansson/stable-diffusion-inpainting instead.",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output. If the NSFW filter is triggered, you may get fewer outputs than this.",
            ge=1,
            le=10,
            default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="EULER-DISCRETE",
            choices=["DDIM", "K-LMS", "PNDM", "EULER-DISCRETE"],
            description="Choose a scheduler. If you use an init image, PNDM will be used. Default: EULER-DISCRETE",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        progress_callback_url: str = Input(
            description="The URL to which a percent will be posted.", default=None
        ),
        image_callback_url: str = Input(
            description="The URL to which each individual step will be POST-ed as a byte stream.", default=None
        ),
        callback_frequency: int = Input(
            description="A post request will be made to the callback url every `callback_frequency` iterations.  Setting this too high will lead to slower generation.", ge=1, default=5
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        extra_kwargs = {}
        if mask:
            if not init_image:
                raise ValueError("mask was provided without init_image")
            pipe = self.inpaint_pipe
            init_image = Image.open(init_image).convert("RGB")
            extra_kwargs = {
                "mask_image": Image.open(mask).convert("RGB").resize(init_image.size),
                "init_image": init_image,
                "strength": prompt_strength,
            }
        elif init_image:
            pipe = self.img2img_pipe
            extra_kwargs = {
                "init_image": Image.open(init_image).convert("RGB"),
                "strength": prompt_strength,
            }
        else:
            pipe = self.txt2img_pipe

        pipe.scheduler = make_scheduler(scheduler, progress_callback_url, image_callback_url, callback_frequency)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            **extra_kwargs,
        )
        num_outputs = len(output)

        samples = [
            output.images[i]
            for i, nsfw_flag in enumerate(output.nsfw_content_detected)
            if not nsfw_flag
        ]

        if len(samples) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        if num_outputs > len(samples):
            print(
                f"NSFW content detected in {num_outputs - len(samples)} outputs, showing the rest {len(samples)} images..."
            )
        output_paths = []
        for i, sample in enumerate(samples):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths


def make_scheduler(
        name="EULER-DISCRETE",
        image_callback_url: str = "",
        progress_callback_url: str = "",
        callback_frequency: int = -1
):
    scheduler = {
        "EULER-DISCRETE": EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="scheduler"
        ),
        "PNDM": PNDMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
        "K-LMS": LMSDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        ),
        "DDIM": DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        ),
    }[name]

    # While callback frequency default to a higher value to avoid weird behavior, if no urls are provided, snap to -1.
    if not image_callback_url and not progress_callback_url:
        callback_frequency = -1

    if callback_frequency >= 0 and (image_callback_url or progress_callback_url):
        image_callback = None
        if image_callback_url:
            def cb(images: List[Image.Image], ) -> None:
                buffer = BytesIO()
                images[0].save(buffer, format="PNG")
                buffer.seek(0)
                encoded = base64.b64encode(buffer.read()).decode("utf-8")
                requests.post(image_callback_url, json={
                    "status": "success",
                    "output": [f"data:image/png;base64,{encoded}"]
                })
                print(f"Callback with image.  Posting to {image_callback_url}")
            image_callback = cb
        hijack_scheduler_step(scheduler, None, image_callback, callback_frequency)

    return scheduler
