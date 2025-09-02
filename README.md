<div align="center">

# 🚀🚀🚀 Improve Diffusion Image Generation Quality using Levenberg-Marquardt-Langevin

We introduce **LML**, an accelerated sampler for diffusion models leveraging the second-order Hessian geometry. Our LML implementation is completely compatible with the **[diffusers](https://github.com/huggingface/diffusers)**.

This repository is the official implementation of the **ICCV 2025** paper:
_"Unleashing High-Quality Image Generation in Diffusion Sampling Using Second-Order Levenberg-Marquardt-Langevin"_ 


> **Fangyikang Wang<sup>1,2</sup>, Hubery Yin<sup>2</sup>, Lei Qian<sup>1</sup>, Yinan Li<sup>1</sup>, Shaobin Zhuang<sup>3,2</sup>, Huminhao Zhu<sup>1</sup>, Yilin Zhang<sup>1</sup>, Yanlong Tang<sup>4</sup>, Chao Zhang<sup>1</sup>, Hanbin Zhao<sup>1</sup>, Hui Qian<sup>1</sup>, Chen Li<sup>2</sup>**
>
> <sup>1</sup>Zhejiang University <sup>2</sup>WeChat Vision, Tencent Inc <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Tencent Lightspeed Studio

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2505.24222-b31b1b.svg)](https://www.arxiv.org/abs/2505.24222)&nbsp;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/zituitui/LML-diffusion-sampler)
[![Github](https://img.shields.io/badge/Github-LML-white)](https://github.com/zituitui/LML-diffusion-sampler)
[![Zhihu](https://img.shields.io/badge/%E7%9F%A5%E4%B9%8E-LML-informational.svg)](https://zhuanlan.zhihu.com/p/1944817402341724362)
[![Xiaohongshu](https://img.shields.io/badge/%E5%B0%8F%E7%BA%A2%E4%B9%A6-LML-ff2442.svg)](https://www.xiaohongshu.com/explore/68b16f1d000000001b033ab1?source=webshare&xhsshare=pc_web&xsec_token=ABQjYHRC_N07F42JR0lCZyjQ9Ty73T8suGIpcG-8AI6Pg=&xsec_source=pc_share)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)&nbsp;

<img src="assets/lml-sd-visual_2_new-1.png" alt="SD Results" style="width: 100%;">

<img src="assets/lml-celeb-visual-1.png" alt="celeb Results" style="width: 70%;">

</div>

## The intuition of our LML diffusion sampler

![anneal](assets/anneal_path.drawio-1.png)
> **Schematic comparison** between our LML method and baselines. While previous works mainly focus on intriguing designs along the annealing path to improve diffusion sampling, they leave operations at specific noise levels to be performed using first-order Langevin. Our approach proposes to leverage the Levenberg-Marquardt approximated Hessian geometry to guide the Langevin update to be more accurate.


![Some edits](assets/newton_algos.drawio-1.png)
> The relation between optimization algorithms and MCMC sampling algorithms. We initially wanted to develop a diffusion sampler utilizing Hessian geometry, following the path of Newton-Langevin dynamics.
However, this approach proved to be highly computationally expensive within the DM context.
Drawing inspiration from the Levenberg-Marquardt method used in optimization, our method incorporates low-rank approximation and damping techniques. This enables us to obtain the Hessian geometry in a computationally affordable manner. Subsequently, we use this approximated Hessian geometry to guide the Langevin updates.

## 👨🏻‍💻 Run the code 
### 1) Get start

* Python 3.8.12
* CUDA 11.7
* NVIDIA A100 40GB PCIe
* Torch 2.0.0
* Torchvision 0.14.0

Please follow **[diffusers](https://github.com/huggingface/diffusers)** to install diffusers.

### 2) Sampling
first, please switch to the root directory.

- #### CIFAR-10 sampling
  For baseline, you can do CIAFR-10 sampling as follows, choose sampler_type within [ddim, pndm, dpm, dpm++, unipc]:
  ```bash
  python3 ./scripts/cifar10.py --test_num 1 --batch_size 1 --num_inference_steps 10  --save_dir YOUR/SAVE/DIR --model_id xx/xx/ddpm_ema_cifar10 --sampler_type ddim
  ```
  For our LML sampler, there is an additional $\lambda$ hyperparameter:
  ```bash
  python3 ./scripts/cifar10.py --test_num 1 --batch_size 1 --num_inference_steps 10  --save_dir YOUR/SAVE/DIR --model_id xx/xx/ddpm_ema_cifar10 --sampler_type dpm_lm --lamb 0.0008
  ```

  For the optimal choice of LML, we have:
  |         | 5 NFEs  | 6 NFEs  | 7 NFEs  | 8 NFEs  | 9 NFEs  | 10 NFEs | 12 NFEs | 15 NFEs | 20 NFEs | 30 NFEs | 50 NFEs | 100 NFEs |
  |---------|---------|---------|---------|---------|---------|----------|----------|----------|----------|----------|----------|-----------|
  | optimal value of lamb     | 0.0008  | 0.0008  | 0.001   | 0.001   | 0.001   | 0.0008   | 0.001    | 0.001    | 0.0005   | 0.0003   | 0.0001   | 0.00005   |



- #### CelebA-HQ sampling
  For baseline:
  ```bash
  python3 ./scripts/celeba.py --test_num 1 --batch_size 1 --num_inference_steps 10  --save_dir YOUR/SAVE/DIR --model_id xx/xx/ldm-celebahq-256 --sampler_type ddim
  ```

  For our LML:
  ```bash
  python3 ./scripts/celeba.py --test_num 1 --batch_size 1 --num_inference_steps 10  --save_dir YOUR/SAVE/DIR --model_id xx/xx/ldm-celebahq-256 --sampler_type ddim_lm --lamb 0.005
  ```

  - #### SD-15 and SD-2b on MS-COCO sampling
  ```bash
  python3 ./scripts/StableDiffusion_COCO.py --test_num 30002 --num_inference_steps 10  --save_dir YOUR/SAVE/DIR --model_id xx/xx/stable-diffusion-v1-5 --sampler_type dpm_lm --lamb 0.001
  ```

  For the optimal choice of LML on MS-COCO, for NFEs of {5, 6, 7, 8, 9, 10, 12, 15}, we always choose $\lambda = 0.001$:
  <!-- |NFEs| 5    | 6    | 7    | 8    | 9    | 10   | 12   | 15   |
  |---------|------|------|------|------|------|------|------|------|
  | SD-15   | -- | -- | 0.001 | - | - | -| 0.001 | 0.001 |
  | SD-2b   | -- | - | - | - | - | - | - | - | -->


- #### SD-15, SD-2b, SD-XL, and PixArt-$\alpha$ on T2i-compbench sampling
  Before running the scripts, make sure to clone T2I-CompBench repository. Generated images are stored in the directory "save_dir/model/dataset_category/sampler_type/samples".

  For baseline, you can do T2i-compbench sampling as follows, choose sampler_type within [ddim, pndm, dpm, dpm++, unipc] and model within [sd15, sd2_base, sdxl, pixart]:
  ```bash
  python3 ./scripts/StableDiffusion_PixArt_T2i_Sampling.py --dataset_category color --dataset_path PATH/TO/T2I-COMPBENCH --test_num 10 --num_inference_steps 10 --model_dir YOUR/MODEL/DIR --save_dir YOUR/SAVE/DIR --model sd15 --sampler_type ddim
  ```
  For our LML sampler, there is an additional $\lambda$ hyperparameter:
  ```bash
  python3 ./scripts/StableDiffusion_PixArt_T2i_Sampling.py --dataset_category color --dataset_path PATH/TO/T2I-COMPBENCH --test_num 10 --num_inference_steps 10 --model_dir YOUR/MODEL/DIR --save_dir YOUR/SAVE/DIR --model sd15 --sampler_type dpm_lm --lamb 0.006
  ```

- #### Use our LML diffusion sampler with ControlNet

  **canny**
  ```bash
  python3 ./scripts/control_net_canny.py --num_inference_steps 10  --original_image_path /xxx/xxx/data/input_image_vermeer.png --controlnet_dir /xxx/xxx/sd-controlnet-canny --sd_dir /xxx/xxx/stable-diffusion-v1-5  --save_dir YOUR/SAVE/DIR  --sampler_type dpm_lm --lamb 0.001
  ```

  **depth**
  ```bash
  python3 ./scripts/control_net_depth.py --num_inference_steps 10  --controlnet_dir /xxx/xxx/control_v11f1p_sd15_depth --sd_dir /xxx/xxx/stable-diffusion-v1-5  --save_dir YOUR/SAVE/DIR  --sampler_type dpm_lm --lamb 0.001
  ```

  **pose**
  ```bash
  python3 ./scripts/control_net_canny.py --num_inference_steps 10 --controlnet_dir /xxx/xxx/sd-controlnet-openpose --sd_dir /xxx/xxx/stable-diffusion-v1-5  --save_dir YOUR/SAVE/DIR  --sampler_type dpm_lm --lamb 0.001
  ```


- #### LML sampling on FLUX
  For baseline:
  ```bash
  python3 ./scripts/FLUX_T2i_Sampling.py --dataset_category color --dataset_path PATH/TO/T2I-COMPBENCH --test_num 10 --num_inference_steps 10 --model_id YOUR/MODEL/DIR --save_dir YOUR/SAVE/DIR --sampler_type fm_euler
  ```
  For our LML:
  ```bash
  python3 ./scripts/FLUX_T2i_Sampling.py --dataset_category color --dataset_path PATH/TO/T2I-COMPBENCH --test_num 10 --num_inference_steps 10 --model_id YOUR/MODEL/DIR --save_dir YOUR/SAVE/DIR --sampler_type lml_euler --lamb 0.01
  ```


### 3) Evaluation
- #### FID evaluation on CIFAR-10
  [Coming Soon] ⏳

- #### FID evaluation on MS-COCO
  [Coming Soon] ⏳

- #### T2I-compbench evaluation 
  Please refer to the [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) guide. Create a new environment and install the dependencies for T2I-CompBench evaluation.
  For testing combinations of multiple models and samplers, we also provide a convenient one-click script. Place the script file in the corresponding directory of **T2I-CompBench** to replace the origin script. For example:
  ```sh
  # BLIP-VQA for Attribute Binding
  cd T2I-CompBench
  bash BLIPvqa_eval/test.sh
        ||
        ||
        \/
  cp evaluations/T2I-CompBench/BLIPvqa_test.sh T2I-CompBench/BLIPvqa_eval
  cd T2I-CompBench
  bash BLIPvqa_eval/BLIPvqa_test.sh 'save_dir'
  ```
  The directory structure of **'save_dir'** should satisfy the following format:
  ```
  {save_dir}/model/dataset_category/sampler_type/samples/
                                                ├── a green bench and a blue bowl_000000.png
                                                ├── a green bench and a blue bowl_000001.png
                                                └──...
  ```

### 4) Pretrained Diffusion Models
We adopt well-pretrained diffusion models from the community. Thanks for these contributions! Here we list the links to the pretrained diffusion models.

ddpm-ema-cifar10:
     
https://github.com/VainF/Diff-Pruning/releases/download/v0.0.1/ddpm_ema_cifar10.zip

ldm-celebahq-256:
    
https://huggingface.co/CompVis/ldm-celebahq-256
    
stable-diffusion-v1.5:
    
https://huggingface.co/runwayml/stable-diffusion-v1-5

stable-diffusion-v2-base:

https://huggingface.co/stabilityai/stable-diffusion-2-base

stable-diffusion-XL:

https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

PixArt-α:

https://huggingface.co/PixArt-alpha/PixArt-XL-2-512x512



## 🪪 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## 📝 Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{wang2025unleashing,
  title={Unleashing High-Quality Image Generation in Diffusion Sampling Using Second-Order Levenberg-Marquardt-Langevin},
  author={Wang, Fangyikang and Yin, Hubery and Qian, Lei and Li, Yinan and Zhuang, Shaobin and Zhu, Huminhao and Zhang, Yilin and Tang, Yanlong and Zhang, Chao and Zhao, Hanbin and others},
  journal={arXiv preprint arXiv:2505.24222},
  year={2025}
}
```

## 📩 Contact me
Our e-mail address:
```
wangfangyikang@zju.edu.cn, qianlei33@zju.edu.cn, liyinan@zju.edu.cn
```
