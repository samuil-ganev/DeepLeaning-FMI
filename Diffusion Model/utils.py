from parameters import *
from vae import *
from unet import *
from clip import *

from model_loader import load_from_standard_weights


def preload_models_from_standard_weights(ckpt_path, device):

    state_dict = load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP({}).to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        # 'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        # 'diffusion': diffusion
    }


def rescale(tensor, old_range, new_range):
    old_min, old_max = old_range
    new_min, new_max = new_range

    normalized = (tensor - old_min) / (old_max - old_min)

    rescaled = normalized * (new_max - new_min) + new_min

    return rescaled


def get_time_embedding(time_steps, temb_dim = 320):
    assert temb_dim % 2 == 0, "kaput si s vremeto"

    time_steps = time_steps.to(device)

    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=device) / (temb_dim // 2))
    )

    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb.to(device)


def fid_score(img_generated, img):
    from torchmetrics.image.fid import FrechetInceptionDistance
    img = img.to(dtype=torch.uint8, device='cpu')
    fid = FrechetInceptionDistance(feature=64)
    img_generated = img_generated.reshape(3, 3, 224, 224)
    print(img.shape, img_generated.shape)
    fid.update(img, real=True)
    fid.update(img_generated, real=False)
    return fid.compute()