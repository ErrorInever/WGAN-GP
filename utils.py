import torch
import random
import numpy as np
import argparse
import os
import torchvision
from tqdm import tqdm
from matplotlib import pyplot as plt
from data.datasets import AnimeFacesDataset
from torch.utils.data import DataLoader
from config import cfg


# not working if generator has tanh() output, because values go outside [-1, 1], therefore generator limited
def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in tqdm(dataloader, total=len(dataloader)):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


def get_random_noise(size, dim, device):
    """
     Get random noise from normal distribution
    :param size: ``int``, number of samples (batch)
    :param dim: ``int``, dimension
    :param device: cuda or cpu device
    :return: Tensor([size, dim, 1, 1])
    """
    return torch.randn(size, dim, 1, 1).to(device)


def latent_space_interpolation_sequence(latent_seq, step_interpolation=5):
    """
    Interpolation between noises
    :param latent_seq: Tensor([N, z_dim, 1, 1])
    :param step_interpolation: ``int``: number of steps between each images
    :return: List([samples, z_dim, 1, 1]
    """
    vector = []
    alpha_values = np.linspace(0, 1, step_interpolation)

    start_idxs = [i for i in range(0, len(latent_seq))]
    end_idxs = [i for i in range(1, len(latent_seq))]

    for start_idx, end_idx in zip(start_idxs, end_idxs):
        latent_start = latent_seq[start_idx].unsqueeze(0)
        latent_end = latent_seq[end_idx].unsqueeze(0)
        for alpha in alpha_values:
            vector.append(alpha*latent_end + (1.0 - alpha)*latent_start)
    return torch.cat(vector, dim=0)


def gradient_penalty(critic, real, fake, device="cpu"):
    """
    Calculate gradient penalty
    :param critic: Critic()
    :param real: real batch
    :param fake: fake batch
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

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


def checkpoint(epoch, end_epoch, gen, critic, opt_gen, opt_critic, fixed_noise):
    print("=> Saving checkpoint")
    torch.save({
        'gen': gen.state_dict(),
        'critic': critic.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_critic': opt_critic.state_dict(),
        'start_epoch': epoch,
        'end_epoch': end_epoch,
        'fixed_noise': fixed_noise
    }, f"{cfg.SAVE_CHECKPOINT_PATH}/WGAN_GP_epoch_{epoch}.pth.tar")
    print(f"=> Checkpoint save to {cfg.SAVE_CHECKPOINT_PATH}")


def load_checkpoint(checkpoint, gen, critic, opt_gen, opt_critic):
    print("=> Load checkpoint...")
    gen.load_state_dict(checkpoint['gen'])
    critic.load_state_dict(checkpoint['critic'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_critic.load_state_dict(checkpoint['opt_critic'])
    print("=> Checkpoint loaded")
    return checkpoint['start_epoch'], checkpoint['end_epoch'], checkpoint['fixed_noise']


def load_gen(gen, filename, device):
    print("=> Load generator...")
    cp = torch.load(filename, map_location=device)
    gen.load_state_dict(cp['gen'])
    print(f"=> Generator model loaded from {filename}")


def set_seed(val):
    """
    Freezes random sequences
    :param val: ``int`` random value
    """
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)


def show_batch(batch, save, num_samples=36, figsize=(10, 10), normalize=True):
    """
    Show image
    :param batch: ``Tensor([N, channels, size, size])`` batch
    :param num_samples: ``int``: number of sumples
    :param figsize: ``Tuple(n, n)``: size of image
    :param normalize: if need denormalization
    """
    images = batch[:num_samples, ...]
    nrows = int(np.sqrt(num_samples))
    grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1))
    plt.axis('off')

    if save:
        save_path = os.path.join(save, 'grid_result.png')
        plt.savefig(save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean and std')
    parser.add_argument('--path', dest='path', help='path to image folder of dataset',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.path, 'path not specified'
    dataset = AnimeFacesDataset(args.path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    mean, std = get_mean_std(dataloader)
    print(f"mean:{mean}  std:{std}")
