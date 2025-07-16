import json
import numpy as np
import math
import csv
import random
import argparse
import torch
import os
import torch.distributed as dist

from torchvision import transforms
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, \
    DiffusionPipeline, UNet2DConditionModel, AutoencoderKL, \
    DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler

from torch.nn.parallel import DistributedDataParallel as DDP

from reward_model.eval_pickscore import PickScore
import hpsv2
import ImageReward as RM
from reward_model.aesthetic_scorer import AestheticScorer
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

device = torch.device('cuda')


def get_args():
    # pick: test_unique_caption_zh.csv       draw: drawbench.csv
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt_dataset", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str)

    parser.add_argument("--inference_step", default=50, type=int)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--T_max", default=1, type=int)
    parser.add_argument("--RatioT", default=1.0, type=float)    #if RatioT==1,则退化为start point优化
    parser.add_argument("--denoising_cfg", default=5.5, type=float)
    parser.add_argument("--inversion_cfg", default=1.0, type=float)
    parser.add_argument("--method", default='inversion', type=str, choices=['inversion', 'blender', 'repaint'])

    # AYS inference-step = 10
    parser.add_argument("--model", default=None, choices=['DPO', 'AYS', 'OneMoreStep'], type=str)

    args =  parser.parse_args()
    return args


def load_prompt(path, seed_path, prompt_version):
    if prompt_version == 'pick':
        prompts = []
        with open(path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])

        prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list
    
    elif prompt_version == 'draw':
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)

        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)
        return prompts, seed_list
    
    else:
        prompts = []
        with open(path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])
        # prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list

        #seed
        with open(seed_path) as f:
            seed_list = json.load(f)

        return prompts, seed_list


def load_pick_prompt(path):
    prompts = []
    seeds = []
    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        
        for row in content:
            data = eval(row)
            prompts.append(data['caption'])
            seeds.append(data['seed'])
        
    return prompts, seeds


# PICK_MODEL = PickScore()
HPSV2_MODEL = hpsv2
# IR_MODEL = RM.load("ImageReward-v1.0")
# AES_MODEL = AestheticScorer(dtype = torch.float32)



def cal_score(prompt, image):

    # _, pick_score = PICK_MODEL.calc_probs(prompt, image)

    hpsv2_score = HPSV2_MODEL.score([image], prompt, hps_version="v2.1")[0]

    # ir_score = IR_MODEL.score(prompt, image)

    # aes_score = AES_MODEL(image)

    # return [pick_score.item(), float(hpsv2_score), ir_score, aes_score.item()]
    return [float(hpsv2_score)]


if __name__ == '__main__':

    dtype = torch.float16
    args = get_args()

    
    # load pipe
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                        variant='fp16',
                                                        safety_checker=None, requires_safety_checker=False).to(device)
    
    if not args.model:
        
        # pipe.enable_xformers_memory_efficient_attention()
        # unet = DDP(pipe.unet, device_ids=[local_rank], output_device=local_rank)

        inverse_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                                subfolder='scheduler')
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inv_scheduler = inverse_scheduler


    prompt_list, seed_list = load_pick_prompt(
        path='train_60000.json'
    )

    # mkdir
    base_dir = 'datasets/Output_test'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(os.path.join(base_dir,'origin')):
        os.mkdir(os.path.join(base_dir,'origin'))
    if not os.path.exists(os.path.join(base_dir,'optim')):
        os.mkdir(os.path.join(base_dir,'optim'))
    if not os.path.exists(os.path.join(base_dir,'show')):
        os.mkdir(os.path.join(base_dir,'show'))
    if not os.path.exists(os.path.join(base_dir,'latents')):
        os.mkdir(os.path.join(base_dir,'latents'))

    T_max = args.T_max
    size = args.size
    shape = (1, 4, size // 8, size // 8)
    num_steps = args.inference_step
    guidance_scale = args.denoising_cfg
    inversion_guidance_scale = args.inversion_cfg
    ratioT = args.RatioT

    before_score, after_score, positive = 0, 0, 0

    # -------------------- PickSore Dataset ------------------------------------
    # 3 paths, 2 idx, 3 paths
    with open('SDXL_step_10_training.json', 'w+') as file:
        
        for idx, prompt in enumerate(prompt_list):
            random_seed = seed_list[idx]  # 拿到seed_list中的 seed

            np.random.seed(int(random_seed))
            torch.manual_seed(int(random_seed))
            torch.cuda.manual_seed(int(random_seed))
            generator = torch.manual_seed(random_seed)
            start_latents = torch.randn(shape, generator=generator, dtype=dtype).to(device)

            original_img = pipe(
                prompt=prompt,
                height=size,
                width=size,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                latents=start_latents).images[0]

            # original_img.save(os.path.join(base_dir, 'origin', f'{idx}.png'))

            # Inversion
            optim_img = pipe.forward_ours(
                prompt=prompt,
                height=size,
                width=size,
                guidance_scale=guidance_scale,
                inversion_guidance_scale=inversion_guidance_scale,
                num_inference_steps=num_steps,
                latents=start_latents,
                T_max=T_max,
                index=idx,
                seed=random_seed,
                output_dir=args.output_dir).images[0]  # 多加了一个 index，方便存储数据

            # optim_img.save(os.path.join(base_dir, 'optim', f'{idx}.png'))

            # original_img = optim_img
            # new_width = original_img.width + optim_img.width
            # new_image = Image.new("RGB", (new_width, original_img.height))
            # new_image.paste(original_img, (0, 0))
            # new_image.paste(optim_img, (original_img.width, 0))
            # 保存拼接后的图片
            # new_image.save(os.path.join(base_dir, 'show', f'{idx}.png'))

            original_scores = cal_score(prompt, original_img)
            optimized_scores = cal_score(prompt, optim_img)

            # pick_score --> hpsv2 --> ir --> aes
            data = {
                'index': idx,
                'seed': random_seed,
                'caption': prompt,
                'original_score_list': original_scores,
                'optimized_score_list': optimized_scores
                
            }
            json.dump(data, file)
            file.write('\n')





