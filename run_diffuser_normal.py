from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers.utils import load_image
from diffusers import UniPCMultistepScheduler
import os
import os.path as osp
import numpy as np
from PIL import Image

# controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_xformers_memory_efficient_attention()
pipe.safety_checker = lambda images, clip_input: (images, False) # avoid NSFW detection

num = 4
prompts = ["leather", 'feather', 'wood', 'paper', 'metal']
# https://huggingface.co/blog/controlnet
for p in prompts:
    for n in range(num):
        for i in range(25):
            print(f'{p}: {n}/{num}: {i}/25')
            # fpath = f'/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/depth/cam{i:02d}.png'
            fpath = f'/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/unwrap_normal/combine_test.png'
            depth_image = np.array(load_image(fpath))[::2,::2,:]
            # depth_image = (depth_image / 255.) * 65535.
            
            # cut by bounding box to save computation
            fg = np.where(depth_image)
            hw_min = list(map(lambda r: r.min(), fg[0:2]))
            hw_max = list(map(lambda r: r.max(), fg[0:2]))
            depth_cut = depth_image[hw_min[0]:hw_max[0],hw_min[1]:hw_max[1]]
            
            generator = [torch.Generator(device="cpu").manual_seed(n*1000)]

            output = pipe(
                [p + ', best quality, extremely detailed'],
                image=Image.fromarray(depth_cut),
                # image=torch.FloatTensor(depth_cut),
                negative_prompt=['longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'],
                generator=generator,
                num_inference_steps=20,
            )

            vid = osp.splitext(osp.basename(fpath))[0].split('_')[0]
            path_out = f'./logs_uv_normal/{p}/{n}/{vid}.png'
            os.makedirs(osp.dirname(path_out), exist_ok=True)
            out = np.zeros_like(depth_image)
            out[hw_min[0]:hw_max[0],hw_min[1]:hw_max[1]] = np.array(output.images[0].resize(depth_cut.shape[:2][::-1]))
            Image.fromarray(out).save(path_out)
            # imageio.imwrite(path_out, out)
            # output.images[0].save(path_out)
            break