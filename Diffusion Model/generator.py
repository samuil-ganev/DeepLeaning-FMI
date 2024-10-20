import torch
import numpy as np
from utils import *
from parameters import *

WIDTH = 224
HEIGHT = 224
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def step(model, timestep, latents, model_output):
    t = timestep
    prev_t = timestep - train_steps // inf_steps

    alpha_prod_t = model.alpha_bars[t]
    alpha_prod_t_prev = model.alpha_bars[prev_t] if prev_t >= 0 else torch.tensor(1.0)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # предиктнато x_0 по формула (15) от https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

    # Оценка на 'средния' вектор от многомерното нормално разпределение спрямо x_t и x_0 по формула (7) от https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

    # Оценка на дисперсията от многомерното нормално разпределение по формула (7) от https://arxiv.org/pdf/2006.11239.pdf
    sigma = 0
    if t > 0:
        noise = torch.randn(model_output.shape, device=device, dtype=model_output.dtype)

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)
        sigma = variance ** 0.5 * noise

    # Семплираме от N(mu, sigma) = X като X = mu + sigma * N(0, 1)
    pred_prev_sample = pred_prev_sample + sigma

    return pred_prev_sample


def generate(model, prompt, mode, device = None):
    model.eval()

    print(prompt)

    with torch.no_grad():

        tokens = [prompt.lower().strip().split()]
        context = model.clip(tokens)

        latent_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
        latents = torch.randn(latent_shape, device=device)

        step_ratio = model.n_steps // inf_steps
        timesteps = torch.from_numpy((np.arange(0, inf_steps) * step_ratio).round()[::-1].copy().astype(np.int64))
        # timesteps = (0, 4, 8, 12, ..., 192, 196)

        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep.view(1))
            if mode == 'SD':
                model_output = model.diff(latents, context, time_embedding)
            else:
                model_output = model.vt(latents, context)
            latents = step(model, timestep, latents, model_output)

        image = model.vae_decoder(latents)
        image = rescale(image, (-1, 1), (0, 255))
        # image.shape = (batch, c, h, w)

        image = image.permute(0, 2, 3, 1)
        # image.shape = (batch, h, w, c)

        image = image.to("cpu", torch.uint8).numpy()

        return image[0]
