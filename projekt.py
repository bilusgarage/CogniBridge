import huggingface_hub
huggingface_hub.cached_download = huggingface_hub.hf_hub_download


import mindspore as ms  # <-- Import MindSpore

import mindnlp

from diffusers import DiffusionPipeline
tensor.grad = 1
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ms_dtype=ms.float16
)
image = pipe("A sunset over mountains, oil painting style").images[0]
image.save("sunset.png")