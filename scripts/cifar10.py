import sys
import time
import torch
import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
# __dir__ = os.path.dirname(os.path.abspath('adjoint_state'))
sys.path = [os.path.abspath(os.path.join(__dir__, '../../libs'))] + sys.path

import json
import argparse
sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline, PNDMLMPipeline, DDPMLMPipeline, DPMLMPipeline, UniPCPipeline


def main():
    parser = argparse.ArgumentParser(description="sampling script for COCO14 on chongqing mach_ine.")
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--vindex', type=int, default=8)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=[ 'pndm', 'ddim', 'dpm++_lm', 'dpm++', 'dpm','dpm_lm', 'unipc'])
    # parser.add_argument('--model', type=str, default='sd15', choices=['sd15', 'sd2_base'])
    parser.add_argument('--save_dir', type=str, default='/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/felixfwang/result/0402')
    parser.add_argument('--model_id', type=str,
                        default='/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/felixfwang/ddpm_ema_cifar10')
    parser.add_argument('--bdia_gamma', type=float, default=1.0)
    parser.add_argument('--edict_p', type=float, default=0.93)
    parser.add_argument('--dual_c', type=float, default=0.5)
    parser.add_argument('--lamb', type=float, default=1.0)
    parser.add_argument('--kappa', type=float, default=0.0)
    parser.add_argument('--dtype', type=str,
                        default='fp32')

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

    gamma = args.bdia_gamma
    p = args.edict_p
    start_index = args.start_index
    batch_size = args.batch_size
    sampler_type = args.sampler_type
    test_num = args.test_num
    num_inference_steps = args.num_inference_steps
    vindex = args.vindex
    dual_c = args.dual_c
    lamb = args.lamb
    kappa = args.kappa

    # load model
    model_id = args.model_id
    # model_id = "/mnt/chongqinggeminiceph1fs/geminicephfs/mm-base-vision/felixfwang/ddpm-ema-celebahq-256"
    # ddpm = DDPMPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
    # ddpm.unet.to('cuda')
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for seed in range(start_index,start_index+test_num):
            print('prepare to sample')
            start_time = time.time()
            torch.manual_seed(seed)
            if sampler_type in ['pndm']:
                pipe = PNDMPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.unet.to('cuda')
                images = pipe(batch_size = batch_size, num_inference_steps = num_inference_steps).images
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"pndm_cifar10_inference{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"pndm batch##{seed},done")
            elif sampler_type in ['dpm++']:
                pipe = DPMLMPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.unet.to('cuda')
                pipe.scheduler.config.solver_order = 3
                pipe.scheduler.config.algorithm_type = "dpmsolver++"
                pipe.scheduler.lamb = lamb
                pipe.scheduler.lm = False
                pipe.scheduler.kappa = kappa
                images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"seed{seed}_{i}_dpm++_cifar10_inference{num_inference_steps}.png"))
                print('pipe.scheduler.config.solver_order', pipe.scheduler.config.solver_order)
                print(f"dpm++ batch##{seed},done")
            elif sampler_type in ['dpm_lm']:
                pipe = DPMLMPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.unet.to('cuda')
                pipe.scheduler.config.solver_order = 3
                pipe.scheduler.config.algorithm_type = "dpmsolver"
                pipe.scheduler.lamb = lamb
                pipe.scheduler.lm = True
                pipe.scheduler.kappa = kappa
                images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"seed{seed}_{i}_dpm_lm_cifar10_inference{num_inference_steps}_lamb{lamb}_kappa{kappa}.png"))
                print('pipe.scheduler.config.solver_order', pipe.scheduler.config.solver_order)
                print(f"dpm_lm batch##{seed},done")
            elif sampler_type in ['dpm']:
                pipe = DPMLMPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.unet.to('cuda')
                pipe.scheduler.config.solver_order = 3
                pipe.scheduler.config.algorithm_type = "dpmsolver"
                pipe.scheduler.lamb = lamb
                pipe.scheduler.lm = False
                pipe.scheduler.kappa = kappa
                images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"seed{seed}_{i}_dpm_cifar10_inference{num_inference_steps}.png"))
                print('pipe.scheduler.config.solver_order', pipe.scheduler.config.solver_order)
                print(f"dpm batch##{seed},done")
            elif sampler_type in ['ddim']:
                ddpm = DDPMPipeline.from_pretrained(model_id, torch_dtype=dtype)
                ddpm.unet.to('cuda')
                images = ddim_forward(ddpm_pipe=ddpm, batch_size=batch_size, num_inference_steps=num_inference_steps,  torch_dtype=dtype)
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"seed{seed}_{i}_ddim_cifar10_inference_pipe{num_inference_steps}.png"))
                print(f"ddim batch##{seed},done")
            elif sampler_type in ['unipc']:
                pipe = UniPCPipeline.from_pretrained(model_id, torch_dtype=dtype)
                pipe.unet.to('cuda')
                images = pipe(batch_size=batch_size, num_inference_steps=num_inference_steps).images
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"dpm_cifar10_inference{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"unipc batch##{seed},done")

            end_time = time.time()
            time_difference = end_time - start_time
            print(f"The code took {time_difference} seconds to run.")

if __name__ == '__main__':
    main()

