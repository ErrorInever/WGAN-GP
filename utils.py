import torch
import torch.nn as nn
import os
from config import cfg


def gradient_penalty(critic, labels, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, labels)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_models(epoch, gen, critic):
    print("=> Saving checkpoint")
    torch.save(gen.state_dict(), f'{cfg.SAVE_MODELS_PATH}/generator_st_{epoch}.pkl')
    torch.save(critic.state_dict(), f'{cfg.SAVE_MODELS_PATH}/critic_st_{epoch}.pkl')
    print(f"=> Models save to {cfg.SAVE_MODELS_PATH}/generator_st_{epoch}.pkl & {cfg.SAVE_MODELS_PATH}/critic_st_{epoch}.pkl")


def load_models(gen, critic, generator_filename, critic_filename):
    gen_path = os.path.join(os.getcwd(), generator_filename)
    critic_path = os.path.join(os.getcwd(), critic_filename)
    print("=> Load models...")
    gen.load_state_dict(torch.load(gen_path))
    critic.load_state_dict(torch.load(critic_path))
    print(f"=> Generator model loaded from {gen_path}")
    print(f"=> Critic model loaded from {critic_path}")

def load_gen(gen, filename):
    print("=> Load generator...")
    gen_path = os.path.join(os.getcwd(), filename)
    gen.load_state_dict(torch.load(gen_path))
    print(f"=> Generator model loaded from {gen_path}")