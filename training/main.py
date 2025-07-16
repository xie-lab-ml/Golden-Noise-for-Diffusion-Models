import json
import os
import random
import argparse
import numpy as np
import accelerate
import torch
import torch.distributed as dist

from PIL import Image
from torch import nn
from torch.nn.functional import mse_loss
from diffusers import DDIMScheduler, DDIMInverseScheduler, DPMSolverMultistepScheduler, \
                        Transformer2DModel, BitsAndBytesConfig, SD3Transformer2DModel
from reward_model.eval_pickscore import PickScore
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from language.t5 import T5Embedder
from solver import solver_dict
from noise_dataset import NoiseDataset
from reward_model.eval_pickscore import PickScore
from utils.pipeline_stable_diffusion_xl_copy import StableDiffusionXLPipeline
from utils.pipeline_stable_diffusion_21 import StableDiffusionPipeline
from utils.pipeline_hunyuan import HunyuanDiTPipeline
from utils.pipeline_pixel_sigma import PixArtSigmaPipeline
from utils.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from utils.pipeline_flux import FluxPipeline

DEVICE = torch.device("cuda" if torch.cuda else "cpu")

def get_args():
    parser = argparse.ArgumentParser()

    # ddp mode
    parser.add_argument("--ddp", default=False, type=bool)

    # model and dataset construction
    parser.add_argument('--pipeline', default='SDXL', 
                        choices=['SDXL', 'SD2.1', 'DS-turbo', 'DiT'], type=str)
    parser.add_argument("--model", default='svd_unet+unet',
                        choices=['unet', 'vit', 'svd_unet', 
                                 'svd_unet+unet', 'e_unet', 
                                 'svd_unet+unet+dit', 
                                 ], type=str)
    
    
    parser.add_argument("--benchmark-type", default='pick', choices=['pick', 'draw'], type=str)
    parser.add_argument("--train", default=False, type=bool)
    parser.add_argument("--test", default=False, type=bool)

    # hyperparameters
    parser.add_argument('--postfix', default='_hps_sdxl_step_10_random_noise', type=str)
    parser.add_argument('--acculumate-steps', default=64, type=int)
    parser.add_argument('--pick', default=False, type=bool)
    parser.add_argument('--do-classifier-free-guidance', default=True, type=bool)
    parser.add_argument("--inference-step", default=10, type=int)
    parser.add_argument("--size", default=1024, type=int)
    parser.add_argument("--RatioT", default=1.0, type=float)

    # for dreamershaper and flux is 3.5, remaining is 5.5, hunyaun 5.0, pixelart, sd3.5 4.5
    parser.add_argument("--guidance-scale", default=5.5, type=float)
    parser.add_argument("--guidance-rescale", default=0.0, type=float)
    parser.add_argument("--all-file", default=False, type=bool)
    parser.add_argument("--epochs", default=30, type=int)  # more iterations, less epochs ==> 3
    parser.add_argument("--batch-size", default=64, type=int) # verify ESVD, SVD, EUnet with prompt[0]  
    parser.add_argument("--num-workers", default=16, type=int)
    parser.add_argument("--metric-version", default='PickScore', choices=['PickScore', 'HPS v2', 'AES', 'ImageReward'],
                        type=str)

    # path configuration
    parser.add_argument("--prompt-path",
                        default='./sdxl_step_10_training_seed.json',
                        type=str)
    parser.add_argument("--data-dir",
                        default="./datasets/noise_pairs_SDXL_10_pick_total/",
                        type=str)
    parser.add_argument('--pretrained-path', type=str,
                        default='./checkpoints/SDXL-10')
    parser.add_argument('--save-ckpt-path', type=str,
                        default='./checkpoints/SDXL-10/svd_unet+unet')

    # discard the bad samples
    parser.add_argument("--discard", default=False, type=bool)


    args = parser.parse_args()

    print("generating config:")
    print(f"Config: {args}")
    print('-' * 100)

    return args


if __name__ == '__main__':
    dtype = torch.float16
    args = get_args()

    if args.ddp:
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

    # construct the diffusion models and human perference models
    reward_model = PickScore()

    # stabilityai/stable-diffusion-2-1  Lykon/dreamshaper-xl-v2-turbo stabilityai/sdxl-turbo
    if args.pipeline == 'SDXL':
        pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype,
                                                    variant='fp16',
                                                    safety_checker=None, requires_safety_checker=False).to(DEVICE)
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)


    elif args.pipeline == 'DS-turbo':
        
        pipeline = StableDiffusionXLPipeline.from_pretrained("Lykon/dreamshaper-xl-v2-turbo", torch_dtype=dtype,
                                                        variant='fp16',
                                                        safety_checker=None, requires_safety_checker=False).to(DEVICE)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    
    elif args.pipeline == 'DiT':
            pipeline = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", 
                                          torch_dtype=torch.float16).to(DEVICE)

    else:
        print(f'Pipeline {args.pipeline} doesn`t exist!')
        assert False


    # construct the solver
    try:
        if args.ddp:
            solver = solver_dict[args.model](
                pipeline=pipeline,
                local_rank=local_rank,
                config=args
            )
        else:
            solver = solver_dict[args.model](
                pipeline=pipeline,
                config=args
            )
    except:
        print("Solver does not exist!")
        assert False

    # construct the dataset
    if args.train:
        NoiseDataset_100 = NoiseDataset(
            discard=args.discard,
            pick=args.pick,
            all_file=args.all_file,
            evaluate=args.evaluate,
            data_dir=args.data_dir,
            prompt_path=args.prompt_path)


    
        from sklearn.model_selection import StratifiedShuffleSplit

        labels = [0 for i in range(len(NoiseDataset_100))]
        ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]

        trainset = torch.utils.data.Subset(NoiseDataset_100, train_indices)
        valset = torch.utils.data.Subset(NoiseDataset_100, valid_indices)

        print(f'training set size: {len(trainset)}')
        print(f'validation set size: {len(valset)}')

        if args.ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
            train_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_workers)

        else:
            train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)


        solver.train(
            train_loader,
            val_loader,
            total_epoch=args.epochs,
            save_path=args.save_ckpt_path)

    if args.test:
        random_seed = 120
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        latent = None
        random_latent = torch.randn(1, 4, args.size//8, args.size//8, dtype=dtype).to(DEVICE)

        postfix = args.postfix


        test_prompt = "Trump was shot in the right ear while speaking at a campaign rally in Pennsylvania, but extended his arm and shouted 'Fight'"  # a black man
        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)

        random_latent = torch.randn(1, 4, args.size//8, args.size//8, dtype=dtype).to(DEVICE)

        test_prompt = "full glass of water standing on a mountain"  # a black man
        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)
        random_latent = torch.randn(1, 4, args.size//8, args.size//8, dtype=dtype).to(DEVICE)

        test_prompt = "A sign that says ""Hatsune Miku es real"""  # a black man
        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)
        

        test_prompt = "a girl with long silver hair, she looks 15 old, wearing cute dress, anime-style"  # a black man


        solver.generate(random_latent,
                        latent,
                        reward_model,
                        config=args,
                        prompt=test_prompt,
                        save_postfix=test_prompt.replace(" ", "_") + postfix)

    if args.ddp:
        dist.destroy_process_group()