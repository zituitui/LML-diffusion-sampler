import sys
import time
import torch
import os

import json
import argparse
sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMScheduler, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from scheduler.scheduling_dpmsolver_multistep_lm import DPMSolverMultistepLMScheduler
from scheduler.scheduling_ddim_lm import DDIMLMScheduler

def main():
    parser = argparse.ArgumentParser(description="sampling script for CIFAR-10.")
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=[ 'pndm', 'ddim', 'dpm++', 'dpm','dpm_lm', 'unipc'])
    parser.add_argument('--save_dir', type=str, default='/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/felixfwang/result/0402')
    parser.add_argument('--model_id', type=str,
                        default='/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/felixfwang/ddpm_ema_cifar10')
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=0.0)
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

    start_index = args.start_index
    device = args.device
    batch_size = args.batch_size
    sampler_type = args.sampler_type
    test_num = args.test_num
    num_inference_steps = args.num_inference_steps
    lamb = args.lamb
    kappa = args.kappa
    model_id = args.model_id

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # load pipeline
        pipe = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.unet.to(device)

        # load scheduler
        if sampler_type in ['pndm']:
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        elif sampler_type in ['dpm++']:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver++"
        elif sampler_type in ['dpm_lm']:
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lamb = lamb
            pipe.scheduler.lm = True
            pipe.scheduler.kappa = kappa
        elif sampler_type in ['dpm']:
            pipe.scheduler = DPMSolverMultistepLMScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.config.solver_order = 3
            pipe.scheduler.config.algorithm_type = "dpmsolver"
            pipe.scheduler.lm = False
        elif sampler_type in ['ddim']:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        elif sampler_type in ['unipc']:
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        for seed in range(start_index, start_index + test_num):
            print('prepare to sample')
            start_time = time.time()
            torch.manual_seed(seed)
            
            # sampling process
            images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images

            # store the generated images
            for i, image in enumerate(images):
                image.save(
                    os.path.join(save_dir, f"cifar10_{sampler_type}_inference{num_inference_steps}_seed{seed}_{i}.png"))
            print(f"{sampler_type} batch##{seed},done")

            # output the sampling time-costs
            end_time = time.time()
            time_difference = end_time - start_time
            print(f"The code took {time_difference} seconds to run.")

if __name__ == '__main__':
    main()

