import argparse
import torch
import os
import imageio
from models.model import Generator
from utils import load_gen, get_random_noise, show_batch, latent_space_interpolation_sequence
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description='AnimeFace-WGAN-GP')
    parser.add_argument('--path_ckpt', dest='path_ckpt', help='Path to checkpoint of model', default=None, type=str)
    parser.add_argument('--num_samples', dest='num_samples', help='Number of samples', default=1, type=int)
    parser.add_argument('--steps', dest='steps', help='Number of step interpolation', default=5, type=int)
    parser.add_argument('--device', dest='device', help='cpu or gpu', default=None, type=str)
    parser.add_argument('--out_path', dest='out_path', help='Path to output folder, default=save to project folder',
                        default=None, type=str)
    parser.add_argument('--gif', dest='gif', help='Create gif', default=None, type=bool)
    parser.add_argument('--grid', dest='grid', help='Draw grid of images', default=None, type=bool)
    parser.add_argument('--z_size', dest='z_size', help='The size of latent space, default=128', default=128, type=int)
    parser.add_argument('--img_size', dest='img_size', help='Size of output image', default=6, type=int)
    parser.add_argument('--resize', dest='resize', help='if you want to resize images', default=None, type=int)
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.path_ckpt, 'Path to checkpoint not specified'

    if args.gif and args.num_samples < 2:
        raise ValueError('for GIF num_samples must be greater than 1')

    if not args.out_path:
        out_path = 'WGAN-GP'
    else:
        out_path = args.out_path

    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    gen = Generator(128, 3, 64)
    load_gen(gen, args.path_ckpt, device)
    gen.eval()

    if args.grid:
        noise = get_random_noise(args.num_samples, args.z_size, device)
        print("==> Generate IMAGE GRID...")
        output = gen(noise)
        show_batch(output, out_path, num_samples=args.num_samples, figsize=(args.img_size, args.img_size))
    elif args.gif:
        noise = get_random_noise(args.num_samples, args.z_size, device)
        print("==> Generate GIF...")
        images = latent_space_interpolation_sequence(noise, step_interpolation=args.steps)
        output = gen(images)
        if args.resize and isinstance(args.resize, int):
            print(f"==> Resize images to {args.resize}px")
            output = F.interpolate(output, size=args.resize)

        images = []
        for img in output:
            img = img.detach().permute(1, 2, 0)
            images.append(img.numpy())
        save_img_name = 'result.gif'
        save_path = os.path.join(out_path, save_img_name)
        imageio.mimsave(save_path, images, fps=8)
        print(f'GIF save to {save_path}')
