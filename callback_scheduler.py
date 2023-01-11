from typing import Callable, Optional

from diffusers.schedulers.scheduling_utils import BaseOutput
from PIL import Image

# This is a gross, horrifying hack.
# We can't reliably lean on the schduler to give us the number of steps in the setup, so we have to track it.
_hijacked_scheduler_steps = 0


def hijack_scheduler_step(
        scheduler,
        progress_callback: Optional[Callable],
        image_callback: Optional[Callable],
        callback_frequency: int = 0,
):
    global _hijacked_scheduler_steps
    assert callback_frequency >= 0
    _hijacked_scheduler_steps = 0
    previous_scheduler_step = scheduler.step
    def logging_step(*args, **kwargs):
        global _hijacked_scheduler_steps
        _hijacked_scheduler_steps += 1
        # timestep = kwargs.get('timestep', args[1] or 0)  # Not actually the number of steps in all cases.
        step_output = previous_scheduler_step(*args, **kwargs)  # The key operation

        # Do the logging bit:
        if callback_frequency == 0 or _hijacked_scheduler_steps % callback_frequency == 0:
            if image_callback is not None:
                # This is the latent space, NOT the output space.
                prev_sample = None
                if hasattr(scheduler, 'prev_sample'):
                    prev_sample = scheduler.prev_sample
                if prev_sample is None:
                    if isinstance(step_output, BaseOutput):
                        prev_sample = step_output.prev_sample
                    else:
                        prev_sample = step_output[0]
                image_callback(Image.fromarray(prev_sample[0, :, :, :].cpu().permute(0, 2, 3, 1).numpy()))
            if progress_callback is not None:
                # We do not report percent.  We report steps:
                # num_inference_steps = scheduler.num_inference_steps
                progress_callback(_hijacked_scheduler_steps)

        return step_output
    scheduler.step = logging_step
