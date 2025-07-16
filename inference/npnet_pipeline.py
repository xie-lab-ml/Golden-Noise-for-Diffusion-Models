import random
import torch
import argparse
import torch.nn as nn
import numpy as np

from PIL import Image
from diffusers.models.normalization import AdaGroupNorm
from diffusers import  DDIMScheduler, DPMSolverMultistepScheduler, \
                  DDPMScheduler, StableDiffusionXLPipeline, HunyuanDiTPipeline
                        
                    
from model import NoiseTransformer, SVDNoiseUnet


class NPNet(nn.Module):
      def __init__(self, model_id, pretrained_path=True, device='cuda') -> None:
            super(NPNet, self).__init__()

            assert model_id in ['SDXL', 'DreamShaper', 'DiT']

            self.model_id = model_id
            self.device = device
            self.pretrained_path = pretrained_path

            (
                  self.unet_svd, 
                  self.unet_embedding, 
                  self.text_embedding, 
                  self._alpha, 
                  self._beta
             ) = self.get_model()

      def get_model(self):

            unet_embedding = NoiseTransformer(resolution=128).to(self.device).to(torch.float32)
            unet_svd = SVDNoiseUnet(resolution=128).to(self.device).to(torch.float32)

            if self.model_id == 'DiT':
                  text_embedding = AdaGroupNorm(1024 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32)
            else:
                  text_embedding = AdaGroupNorm(2048 * 77, 4, 1, eps=1e-6).to(self.device).to(torch.float32) 

            
            if '.pth' in self.pretrained_path:
                  gloden_unet = torch.load(self.pretrained_path)
                  unet_svd.load_state_dict(gloden_unet["unet_svd"])
                  unet_embedding.load_state_dict(gloden_unet["unet_embedding"])
                  text_embedding.load_state_dict(gloden_unet["embeeding"])
                  _alpha = gloden_unet["alpha"]
                  _beta = gloden_unet["beta"]

                  print("Load Successfully!")

                  return unet_svd, unet_embedding, text_embedding, _alpha, _beta
            
            else:
                  assert ("No Pretrained Weights Found!")
            

      def forward(self, initial_noise, prompt_embeds):

            prompt_embeds = prompt_embeds.float().view(prompt_embeds.shape[0], -1)
            text_emb = self.text_embedding(initial_noise.float(), prompt_embeds)

            encoder_hidden_states_svd = initial_noise
            encoder_hidden_states_embedding = initial_noise + text_emb

            golden_embedding = self.unet_embedding(encoder_hidden_states_embedding.float())

            golden_noise = self.unet_svd(encoder_hidden_states_svd.float()) + (
                        2 * torch.sigmoid(self._alpha) - 1) * text_emb + self._beta * golden_embedding

            return golden_noise
      

def get_args():
      parser = argparse.ArgumentParser()

      # model and dataset construction
      parser.add_argument('--pipeline', default='SDXL', 
                        choices=['SDXL', 'DreamShaper', 'DiT'], type=str)
      parser.add_argument('--prompt', default='A banana on the left of an apple.', type=str)
      parser.add_argument("--inference-step", default=50, type=int)

      # for dreamershaper is 3.5, remaining is 5.5, DiT is 5.0
      parser.add_argument("--cfg", default=5.5, type=float)

      # model pretrained weight path
      parser.add_argument('--pretrained-path', type=str,
                        default='xxx')

      parser.add_argument("--size", default=1024, type=int)

      args = parser.parse_args()

      print("generating config:")
      print(f"Config: {args}")
      print('-' * 100)

      return args


def main(args):
      dtype = torch.float16
      device = torch.device('cuda')

      if args.pipeline == 'SDXL':

            pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                            variant="fp16",use_safetensors=True,
                                                            torch_dtype=torch.float16).to(device)
            
      elif args.pipeline == 'DreamShaper':
            pipe = StableDiffusionXLPipeline.from_pretrained("lykon/dreamshaper-xl-v2-turbo",
                                                            torch_dtype=torch.float16, 
                                                            variant="fp16").to(device)
      
      else:
             pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", 
                                                            torch_dtype=torch.float16).to(device)
             
      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.enable_model_cpu_offload()

      # create the initial noise
      latent = torch.randn(1, 4, 128, 128, dtype=dtype).to(device)


      # use the pre-trained text encoder in T2I models to encode prompts
      prompt_embeds, _, _, _= pipe.encode_prompt(prompt=args.prompt, device=device)

      # create NPNet to get the target noise
      npn_net = NPNet(args.pipeline, args.pretrained_path)

      golden_noise = npn_net(latent, prompt_embeds)

      # standard inference pipeline
      latent = latent.half()
      golden_noise = golden_noise.half()

      pipe = pipe.to(torch.float16)

      standard_img = pipe(
            prompt=args.prompt,
            height=args.size,
            width=args.size,
            num_inference_steps=args.inference_step,
            guidance_scale=args.cfg,
            latents=latent).images[0]
      
      golden_img = pipe(
            prompt=args.prompt,
            height=args.size,
            width=args.size,
            num_inference_steps=args.inference_step,
            guidance_scale=args.cfg,
            latents=golden_noise).images[0]
      
      # image save path
      standard_img.save(f"{args.pipeline}_{args.prompt}_standard_image.jpg")
      golden_img.save(f"{args.pipeline}_{args.prompt}_golden_image.jpg")
      

if __name__ == '__main__':
      args = get_args()
      main(args)
      