import base64
import os
from io import BytesIO
from typing import List

import requests
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from PIL import Image

from callback_scheduler import hijack_scheduler_step


MODEL_ID = "stabilityai/stable-diffusion-2-1"
MODEL_CACHE = "diffusers-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
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
        prompt_strength: float = Input(
            description="Prompt strength when using init image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        progress_callback_url: str = Input(
            description="The URL to which a step_num will be posted.", default=None
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
        
        # Spiteful forced integer cast.  Sometimes width is coming back as 'FieldInfo' instead of int.
        width = int(width)
        height = int(height)

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config, progress_callback_url, image_callback_url, callback_frequency)

        generator = torch.Generator("cuda").manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        # TODO: Disable the NSFW detector to avoid wasting cycles...
        output_paths = []
        for i, sample in enumerate(output.images):
            #if output.nsfw_content_detected and output.nsfw_content_detected[i]:
            #    continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(
        name: str ="EULER-DISCRETE",
        config: dict = {},
        image_callback_url: str = "",
        progress_callback_url: str = "",
        callback_frequency: int = -1
):
    scheduler = {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        #"EULER-DISCRETE": EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler),
        #"PNDM": PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
        #"K-LMS": LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
        #"DDIM": DDIMScheduler(beta_start=0.00085,beta_end=0.012,beta_schedule="scaled_linear",clip_sample=False,set_alpha_to_one=False,),
    }[name]

    # While callback frequency default to a higher value to avoid weird behavior, if no urls are provided, snap to -1.
    if not image_callback_url and not progress_callback_url:
        callback_frequency = -1

    if callback_frequency >= 0 and (image_callback_url or progress_callback_url):
        image_callback = None
        progress_callback = None
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
        if progress_callback_url:
            def cb(step: int, ) -> None:
                requests.post(progress_callback_url, json={"step": step})
            progress_callback = cb
        hijack_scheduler_step(scheduler, progress_callback, image_callback, callback_frequency)

    return scheduler
