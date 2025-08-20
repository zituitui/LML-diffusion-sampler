import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())

from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob


from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import StableDiffusionPipeline, DDPMPipeline, DDIMPipeline, PNDMPipeline, PNDMLMPipeline, DDPMLMPipeline, DPMLMPipeline, UniPCPipeline, LDMPipeline, PNDMScheduler, UniPCMultistepScheduler,DDIMScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

import cv2
import numpy as np



def main():
    parser = argparse.ArgumentParser(description="sampling script for ControlNet-canny on chongqing machine.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--sampler_type', type = str,default='lag')
    parser.add_argument('--prompt', type=str, default='an asian girl')
    parser.add_argument('--original_image_path', type=str, default="/xxx/xxx/data/input_image_vermeer.png")
    parser.add_argument('--lamb', type=float, default=5.0)
    parser.add_argument('--kappa', type=float, default=0.0)
    parser.add_argument('--freeze', type=float, default=0.0)
    # parser.add_argument('--prompt_list', nargs='+', type=str,
    #                     default=['an asian girl'])
    parser.add_argument('--save_dir', type=str, default='/xxx/xxx/result/0402')
    parser.add_argument('--controlnet_dir', type=str, default="/xxx/xxx/sd-controlnet-canny")
    parser.add_argument('--sd_dir', type=str, default="/xxx/xxx/stable-diffusion-v1-5")



    args = parser.parse_args()
    if args.sampler_type in ['bdia']:
        parser.add_argument('--bdia_gamma', type=float, default=0.5)
    if args.sampler_type in ['edict']:
        parser.add_argument('--edict_p', type=float, default=0.93)
    args = parser.parse_args()
    device = 'cuda'
    sampler_type = args.sampler_type
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    lamb = args.lamb
    freeze = args.freeze
    kappa = args.kappa

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16,use_safetensors=True)

    control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_dir, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )
    control_pipe.enable_model_cpu_offload()
    control_pipe.safety_checker = None

    if sampler_type in ['dpm_lm']:
        control_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.scheduler.config.solver_order = 3
        control_pipe.scheduler.config.algorithm_type = "dpmsolver"
        control_pipe.scheduler.lamb = lamb
        control_pipe.scheduler.lm = True
    elif sampler_type in ['dpm']:
        control_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.scheduler.config.solver_order = 3
        control_pipe.scheduler.config.algorithm_type = "dpmsolver"
        control_pipe.scheduler.lamb = lamb
        control_pipe.scheduler.lm = False
    elif sampler_type in ['dpm++']:
        control_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.scheduler.config.solver_order = 3
        control_pipe.scheduler.config.algorithm_type = "dpmsolver++"
        control_pipe.scheduler.lamb = lamb
        control_pipe.scheduler.lm = False
    elif sampler_type in ['dpm++_lm']:
        control_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.scheduler.config.solver_order = 3
        control_pipe.scheduler.config.algorithm_type = "dpmsolver++"
        control_pipe.scheduler.lamb = lamb
        control_pipe.scheduler.lm = True
    elif sampler_type in ['pndm']:
        control_pipe.scheduler = PNDMScheduler.from_config(control_pipe.scheduler.config)
    elif sampler_type in ['ddim']:
        control_pipe.scheduler = DDIMScheduler.from_config(control_pipe.scheduler.config)
    elif sampler_type in ['ddim_lm']:
        control_pipe.scheduler = DDIMLMScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.scheduler.lamb = lamb
        control_pipe.scheduler.lm = True
        control_pipe.scheduler.kappa = kappa
        control_pipe.scheduler.freeze = freeze
    elif sampler_type in ['unipc']:
        control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)

    original_image = load_image(
        args.original_image_path
    )
    image = np.array(original_image)
    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)


    for prompt, negative_prompt in [['the mona lisa',''],
                                    ['an asian girl',''],
                                    ['an asian princess',''],
                                    ['a portrait of a beautiful woman standing amidst a bed of vibrant tulips.',''],
                                    ['a stunning Arabic woman dressed in traditional clothing',''],
                                    ['a stunning Asian woman dressed in traditional clothing',''],
                                    ['a stunning Black woman dressed in traditional clothing', ''],
                                    ['a stunning German woman dressed in traditional clothing', ''],
                                    ['a stunning Japan woman dressed in traditional clothing', ''],
                                    ['a stunning Chinese woman dressed in traditional clothing', ''],
                                    ['a stunning Jewish woman dressed in traditional clothing', ''],
                                    ]:
        for seed in range(1):
            torch.manual_seed(seed)
            res = control_pipe(
                prompt=prompt, negative_prompt=negative_prompt, image=canny_image,num_inference_steps=num_inference_steps,
            ).images[0]

            res.save(os.path.join(save_dir,
                                  f"{args.model}_{prompt[:20]}_seed{seed}_{sampler_type}_infer{num_inference_steps}_g{guidance_scale}_lamb{args.lamb}.png"))



if __name__ == '__main__':
    main()