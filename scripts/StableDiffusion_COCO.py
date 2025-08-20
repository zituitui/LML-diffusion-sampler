import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())

from diffusers import StableDiffusionPipeline, DPMSolverMultistepLMScheduler, DDIMLMScheduler, PNDMScheduler, UniPCMultistepScheduler

from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

def main():
    parser = argparse.ArgumentParser(description="sampling script for COCO14.")
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--sampler_type', type = str, default='ddim')
    parser.add_argument('--model_id', type=str, default='/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/hubery/4_models/pre_models/stable-diffusion-v1-5')
    parser.add_argument('--save_dir', type=str, default='mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/felixfwang/result/0402')
    parser.add_argument('--lamb', type=float, default=5.0)
    parser.add_argument('--kappa', type=float, default=0.0)
    parser.add_argument('--device', type=str, default='cuda')


    args = parser.parse_args()

    start_index = args.start_index
    sampler_type = args.sampler_type
    test_num = args.test_num
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    lamb = args.lamb
    kappa = args.kappa
    device = args.device
    model_id = args.model_id


    # load model
    sd_pipe = None

    sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
    sd_pipe = sd_pipe.to(device)
    print("sd model loaded")
    

    if sampler_type in ['dpm_lm']:
        sd_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.scheduler.config.solver_order = 3
        sd_pipe.scheduler.config.algorithm_type = "dpmsolver"
        sd_pipe.scheduler.lamb = lamb
        sd_pipe.scheduler.lm = True
    elif sampler_type in ['dpm']:
        sd_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.scheduler.config.solver_order = 3
        sd_pipe.scheduler.config.algorithm_type = "dpmsolver"
        sd_pipe.scheduler.lamb = lamb
        sd_pipe.scheduler.lm = False
    elif sampler_type in ['dpm++']:
        sd_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.scheduler.config.solver_order = 3
        sd_pipe.scheduler.config.algorithm_type = "dpmsolver++"
        sd_pipe.scheduler.lamb = lamb
        sd_pipe.scheduler.lm = False
    elif sampler_type in ['dpm++_lm']:
        sd_pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.scheduler.config.solver_order = 3
        sd_pipe.scheduler.config.algorithm_type = "dpmsolver++"
        sd_pipe.scheduler.lamb = lamb
        sd_pipe.scheduler.lm = True
    elif sampler_type in ['pndm']:
        sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)
    elif sampler_type in ['ddim']:
        sd_pipe.scheduler = DDIMLMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.scheduler.lamb = lamb
        sd_pipe.scheduler.lm = False
        sd_pipe.scheduler.kappa = kappa
    elif sampler_type in ['ddim_lm']:
        sd_pipe.scheduler = DDIMLMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.scheduler.lamb = lamb
        sd_pipe.scheduler.lm = True
        sd_pipe.scheduler.kappa = kappa
    elif sampler_type in ['unipc']:
        sd_pipe.scheduler = UniPCMultistepScheduler.from_config(sd_pipe.scheduler.config)

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # COCO prompts
    with open('/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/pazelzhang/make_dataset/fid_3W_json.json') as fr:
        COCO_prompts_dict = json.load(fr)
    image_id = COCO_prompts_dict.keys()
    with torch.no_grad():
        for pi, key in enumerate(image_id):
            if pi >= start_index  and pi < start_index + test_num:
                print(key)
                print(COCO_prompts_dict[key])
                prompt = COCO_prompts_dict[key]
                negative_prompt = None

                for seed in [1]:
                    generator = torch.Generator(device='cuda')
                    generator = generator.manual_seed(args.seed)
                    res = sd_pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, generator=generator).images[0]
                    res.save(os.path.join(save_dir, f"{pi:05d}_{key}_guidance{guidance_scale}_inference{num_inference_steps}_seed{seed}_{sampler_type}.jpg"))
                    print(f"{sampler_type}##{key},done")


if __name__ == '__main__':
    main()