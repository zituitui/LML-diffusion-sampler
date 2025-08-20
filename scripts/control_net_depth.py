import sys

import torch
import os
import argparse
sys.path.append(os.getcwd())

from PIL import Image
import torchvision.transforms as transforms
import numpy as np


from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers import PNDMScheduler, UniPCMultistepScheduler,DDIMScheduler
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

from transformers import pipeline
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="sampling script for ControlNet-depth on chongqing machine.")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--sampler_type', type = str,default='lag')
    parser.add_argument('--model', type=str, default='sd2_base', choices=['sd15', 'sd2_base'])
    parser.add_argument('--prompt', type=str, default='an asian girl')
    parser.add_argument('--lamb', type=float, default=5.0)
    parser.add_argument('--kappa', type=float, default=0.0)
    parser.add_argument('--freeze', type=float, default=0.0)
    parser.add_argument('--prompt_list', nargs='+', type=str,
                        default=['an asian girl'])
    parser.add_argument('--save_dir', type=str, default='/xxx/xxx/result/0402')
    parser.add_argument('--controlnet_dir', type=str, default="/xxx/xxx/control_v11f1p_sd15_depth")
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

    # torch.manual_seed(args.seed)
    controlnet = ControlNetModel.from_pretrained(args.controlnet_dir, torch_dtype=torch.float16, use_safetensors=True)

    control_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.sd_dir,
        controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
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
        # control_pipe.scheduler.lamb = lamb
        # control_pipe.scheduler.lm = False
        # control_pipe.scheduler.kappa = kappa
    elif sampler_type in ['ddim_lm']:
        control_pipe.scheduler = DDIMLMScheduler.from_config(control_pipe.scheduler.config)
        control_pipe.scheduler.lamb = lamb
        control_pipe.scheduler.lm = True
        control_pipe.scheduler.kappa = kappa
        control_pipe.scheduler.freeze = freeze
    elif sampler_type in ['unipc']:
        control_pipe.scheduler = UniPCMultistepScheduler.from_config(control_pipe.scheduler.config)

    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet-img2img.jpg"
    )

    def get_depth_map(image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map

    depth_estimator = pipeline("depth-estimation")
    depth_map = get_depth_map(image, depth_estimator).unsqueeze(0).half().to("cuda")
    transforms.ToPILImage()(depth_map[0]).save(os.path.join(save_dir,
                                  f"depth_map.png"))


    for prompt, negative_prompt in [["lego batman and robin",''],
                                    ["Spider-Man and Superman", ''],
                                    ["A girl and a boy", ''],
                                    ["asian woman and asian man", ''],
                                    ["American Indian woman and American Indian man", ''],
                                    ["A girl and a girl, monalisa style", ''],
                                    ["Elsa and Anna, in the movie Frozen", ''],
                                    ["A woman and a man, wearing suit", ''],
                                    ]:
        for seed in range(20):
            torch.manual_seed(seed)
            res = control_pipe(
                prompt = prompt, image=image, control_image=depth_map,num_inference_steps=num_inference_steps,
            ).images[0]

            res.save(os.path.join(save_dir,
                                  f"{prompt[:20]}_seed{seed}_{sampler_type}_infer{num_inference_steps}_g{guidance_scale}_lamb{args.lamb}.png"))



if __name__ == '__main__':
    main()