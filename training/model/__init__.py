from .NoiseUnet import NoiseUnet
from .NoiseTransformer import NoiseTransformer
from .SVDNoiseUnet import SVDNoiseUnet, SVDNoiseUnet_wo_attention


model_dict = {
      'unet': NoiseUnet,
      'vit': NoiseTransformer,
      'svd_unet': SVDNoiseUnet,
      'svd_unet_wo_attention': SVDNoiseUnet_wo_attention,
      'svd_unet+unet': [SVDNoiseUnet, NoiseUnet],
      'svd_unet+unet+dit': [SVDNoiseUnet, NoiseUnet],
}