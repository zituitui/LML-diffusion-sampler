import sys

import torch
import os
import json
import argparse

sys.path.append(os.getcwd())

from diffusers import StableDiffusionPipeline, PNDMScheduler, UniPCMultistepScheduler, DDIMScheduler, DiffusionPipeline, PixArtAlphaPipeline
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="sampling script for T2I-Bench.")
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--num_inference_steps', type=int, default=10)
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--sampler_type', type = str, default='dpm_lm')
    parser.add_argument('--model', type=str, default='sd15', choices=['sd15', 'sd2_base', 'sdxl', 'pixart'])
    parser.add_argument('--model_dir', type=str, default='XXX')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--lamb', type=float, default=5.0)
    parser.add_argument('--kappa', type=float, default=0.0)
    parser.add_argument('--freeze', type=float, default=0.0)
    parser.add_argument('--dataset_category', type=str, default="color")
    parser.add_argument('--dataset_path', type=str, default="../T2I-CompBench-main")
    parser.add_argument('--dtype', type=str, default='fp32')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    dtype = None
    if args.dtype in ['fp32']:
        dtype = torch.float32
    elif args.dtype in ['fp64']:
        dtype = torch.float64
    elif args.dtype in ['fp16']:
        dtype = torch.float16
    elif args.dtype in ['bf16']:
        dtype = torch.bfloat16

    device = args.device
    start_index = args.start_index
    sampler_type = args.sampler_type
    test_num = args.test_num
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    lamb = args.lamb
    freeze = args.freeze
    kappa = args.kappa
    model_dir = args.model_dir

    # load model
    sd_pipe = None
    if args.model in ['sd15']:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype, safety_checker=None)
        sd_pipe = sd_pipe.to(device)
        print("sd-1.5 model loaded")
    elif args.model in ['sd2_base']:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype, safety_checker=None)
        sd_pipe = sd_pipe.to(device)
        print("sd-2-base model loaded")
    elif args.model in ['sdxl']:
        sd_pipe = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype, safety_checker=None)
        sd_pipe = sd_pipe.to(device)
        print("sd-xl-base model loaded")
    elif args.model in ['pixart']:
        sd_pipe = PixArtAlphaPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype, safety_checker=None)
        sd_pipe = sd_pipe.to(device)
        print("PixArt-XL-2-512x512 model loaded")

    SAMPLER_CONFIG = {
        'dpm_lm': {
            'scheduler': DPMSolverMultistepLMScheduler,
            'params': {'solver_order': 3, 'algorithm_type': "dpmsolver", 'lm': True, 'lamb': lamb, 'kappa': kappa, 'freeze': freeze}
        },
        'dpm': {
            'scheduler': DPMSolverMultistepLMScheduler,
            'params': {'solver_order': 3, 'algorithm_type': "dpmsolver", 'lm': False}
        },
        'dpm++': {
            'scheduler': DPMSolverMultistepLMScheduler,
            'params': {'solver_order': 3, 'algorithm_type': "dpmsolver++", 'lm': False}
        },
        'dpm++_lm': {
            'scheduler': DPMSolverMultistepLMScheduler,
            'params': {'solver_order': 3, 'algorithm_type': "dpmsolver++", 'lm': True, 'lamb': lamb, 'kappa': kappa, 'freeze': freeze}
        },
        'pndm': {'scheduler': PNDMScheduler, 'params': {}},
        'ddim': {'scheduler': DDIMScheduler, 'params': {}},
        'ddim_lm': {
            'scheduler': DDIMLMScheduler,
            'params': {'lm': True, 'lamb': lamb, 'kappa': kappa, 'freeze': freeze}
        },
        'unipc': {'scheduler': UniPCMultistepScheduler, 'params': {}},
    }

    if sampler_type in SAMPLER_CONFIG:
        config = SAMPLER_CONFIG[sampler_type]
        scheduler_class = config['scheduler']
        sd_pipe.scheduler = scheduler_class.from_config(sd_pipe.scheduler.config)
        
        for param, value in config['params'].items():
            if hasattr(sd_pipe.scheduler, param):
                setattr(sd_pipe.scheduler, param, value)
            elif hasattr(sd_pipe.scheduler.config, param):
                 setattr(sd_pipe.scheduler.config, param, value)
    else:
        raise ValueError(f"invalid: '{sampler_type}'.")
        
    save_dir = args.save_dir
    
    if sampler_type in ['ddim_lm', 'dpm++_lm', 'dpm_lm']:
        save_dir = os.path.join(save_dir, args.model, args.dataset_category, sampler_type + "_lambda_" + str(lamb))
    else:
        save_dir = os.path.join(save_dir, args.model, args.dataset_category, sampler_type)
    save_dir = os.path.join(save_dir, "samples")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    def getT2IDataset(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    yield stripped_line
    
    # T2I prompts
    dataset_path = os.path.join(args.dataset_path, 'examples/dataset', args.dataset_category + '_val.txt')
    count = 0
    with tqdm(total=300 * test_num, desc="Generating Images") as pbar:
        try:
            for prompt in getT2IDataset(dataset_path):
                for seed in range(start_index, start_index + test_num):
                    torch.manual_seed(seed)
                    res = sd_pipe(prompt=prompt, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, generator=None).images[0]
                    res.save(os.path.join(save_dir, f"{prompt}_{count:06d}.png"))
                    count += 1
                    pbar.update(1)
        except FileNotFoundError:
            print(f"dataset can not be found: {dataset_path}")
        except Exception as e:
            print(f"unknown error: {str(e)}")
    print(f"{dataset_path} finish")

if __name__ == '__main__':
    main()