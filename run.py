from share import *
import config

import os
import os.path as osp
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_midas = MidasDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
# model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
config.save_memory = True


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        # input_image = HWC3(input_image)
        # detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        # detected_map = HWC3(detected_map)
        # img = resize_image(input_image, image_resolution)
        # H, W, C = img.shape
        
        # detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        padding_h = round(image_resolution / 64) * 64 - image_resolution
        # detected_map = np.vstack((input_image, np.zeros((padding_h, input_image.shape[-1]), dtype=input_image.dtype)))
        detected_map = input_image
        detected_map = HWC3(detected_map)
        H, W, C = detected_map.shape

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # touching model.control_scales only will not really make any differences
        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # model.control_scales = [strength] * 13  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i][:-padding_h] if padding_h > 0 else x_samples[i] for i in range(num_samples)]
    return [detected_map[:-padding_h] if padding_h > 0 else detected_map] + results

if __name__ == '__main__':
    
    fpaths = [
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam00_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam01_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam02_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam03_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam04_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam05_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam06_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam07_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam08_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/cam09_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/0+1000_test_render_depth.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/0_test_render_depth_half.jpg',
        # '/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/0_test_render_depth_half_half.jpg',
        '/home/ICT2000/jyang/projects/ControlNet/input/Suzanne.png',
    ]
    # fpaths = []
    # for i in range(25):
    #     fpaths.append(f'/home/ICT2000/jyang/projects/ObjectReal/logs/v0.4/pine_env_0_texnet_monkey/init/depth/cam{i:02d}.jpg')
    
    for i, fpath in enumerate(fpaths):
        # input_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)[...,0]
        input_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED) / 255.
        # input_image = ((input_image[...,:3] * input_image[...,-1:] + (1-input_image[...,-1:])) * 255).astype(np.uint8)
        input_image = ((np.ones_like(input_image[...,:3]) * np.ceil(input_image[...,-1:])) * 255).astype(np.uint8)
        input_image = cv2.Canny(cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY), 100, 200)
        prompt = 'leather'
        a_prompt = 'best quality, extremely detailed'
        n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        num_samples = 8 # img number
        # image_resolution = 512
        image_resolution = input_image.shape[0]
        detect_resolution = 384
        ddim_steps = 20
        guess_mode = False # default False
        strength = 1.0
        scale = 9.0
        seed = 1908899563
        eta = 0.0
        
        ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
        rets = process(*ips)
        
        vid = osp.splitext(osp.basename(fpath))[0].split('_')[0]
        for i, ret in enumerate(rets):
            if i == 0: continue
            path_out = f'./logs/{prompt}/{i}/{vid}.jpg'
            os.makedirs(osp.dirname(path_out), exist_ok=True)
            cv2.imwrite(path_out, cv2.cvtColor(ret, cv2.COLOR_BGR2RGB))
        # grid = np.concatenate((
        #     np.concatenate((rets[0], rets[1], rets[2]), axis=1), 
        #     np.concatenate((rets[3], rets[4], rets[5]), axis=1), 
        #     np.concatenate((rets[6], rets[7], rets[8]), axis=1)), axis=0)
        # grid = np.concatenate((
        #     np.concatenate((rets[0], rets[1]), axis=1), 
        #     np.concatenate((rets[2], rets[3]), axis=1)), axis=0)
        # cv2.imwrite(f'./logs/{prompt}/{vid}_grid.jpg', cv2.cvtColor(grid, cv2.COLOR_BGR2RGB))