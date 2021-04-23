import time
import torch
import torch.nn as nn
import argparse
import wandb
import torch.optim as optim
from config import cfg
from data.datasets import AnimeFacesDataset
from torch.utils.data import DataLoader
from models.model import Generator, Critic
from metriclogger import MetricLogger
from utils import checkpoint, load_checkpoint, set_seed, get_random_noise, gradient_penalty


def parse_args():
    parser = argparse.ArgumentParser(description='Anime-WGAN-GP')
    parser.add_argument('--data_path', dest='data_path', help='path to dataset folder', default=None, type=str)
    parser.add_argument('--seed', dest='seed', help='seed value', default=None, type=int)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='path to checkpoint.pth.tar', default=None, type=str)
    parser.add_argument('--out_path', dest='out_path', help='path to output folder', default=None, type=str)
    parser.add_argument('--resume_id', dest='resume_id', help='wandb init id for resume metric', default=None, type=str)
    parser.add_argument('--device', dest='device', help='use device, can be - cpu, cuda, tpu', default='cpu', type=str)
    parser.print_help()
    return parser.parse_args()


def epoch_time(f):
    """Calculate time of each epoch"""
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("epoch time: %2.1f min" % ((te-ts)/60))
        return result
    return timed


@epoch_time
def train_one_epoch(epoch, dataloader, gen, critic, opt_gen, opt_critic,
                    fixed_noise, device, metric_logger, num_samples, freq=100):
    """
    Train one epoch
    :param epoch: ``int`` current epoch
    :param dataloader: object of dataloader
    :param gen: Generator model
    :param critic: Discriminator model
    :param opt_gen: Optimizer for generator
    :param opt_critic: Optimizer for discriminator
    :param fixed_noise: ``tensor[[cfg.BATCH_SIZE, latent_space_dimension, 1, 1]]`` fixed noise (latent space) for image metrics
    :param device: cuda device or cpu
    :param metric_logger: object of MetricLogger
    :param num_samples: ``int`` well retrievable sqrt() (for example: 4, 16, 64) for good result,
    number of samples for grid image metric
    :param freq: ``int``, freq < len(dataloader)`` freq for display results
    """
    for batch_idx, img in enumerate(dataloader):
        real = img.to(device)
        # Train critic
        acc_real = 0
        acc_fake = 0
        for _ in range(cfg.CRITIC_ITERATIONS):
            noise = get_random_noise(cfg.BATCH_SIZE, cfg.Z_DIMENSION, device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            acc_real += critic_real
            critic_fake = critic(fake).reshape(-1)
            acc_fake += critic_fake
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = -1 * (torch.mean(critic_real) - torch.mean(critic_fake)) + cfg.LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        acc_real = acc_real / cfg.CRITIC_ITERATIONS
        acc_fake = acc_fake / cfg.CRITIC_ITERATIONS

        # Train generator: minimize -E[critic(gen_fake)]
        output = critic(fake).reshape(-1)
        loss_gen = -1 * torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # logs metrics
        if batch_idx % freq == 0:
            with torch.no_grad():
                metric_logger.log(loss_critic, loss_gen, acc_real, acc_fake)
                fake = gen(fixed_noise)
                metric_logger.log_image(fake, num_samples, epoch, batch_idx, len(dataloader))
                metric_logger.display_status(epoch, cfg.NUM_EPOCHS, batch_idx, len(dataloader),
                                             loss_critic, loss_gen, acc_real, acc_fake)


if __name__ == '__main__':
    args = parse_args()
    assert args.data_path, 'dataset not specified'

    if args.out_path:
        cfg.OUT_DIR = args.out_path
        cfg.SAVE_CHECKPOINT_PATH = args.out_path
    # set random seed
    if args.seed:
        set_seed(args.seed)
    else:
        set_seed(7889)

    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = "cuda"
    elif args.device == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()

    print(f"=> Called with args {args.__dict__}")
    print(f"=> Config params {cfg.__dict__}")
    print(f"=> Run on device {device}")
    # define dataset and dataloader
    dataset = AnimeFacesDataset(args.data_path)
    cfg.DATASET_SIZE = len(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True,
                            pin_memory=True)
    # define models
    gen = Generator(cfg.Z_DIMENSION, cfg.CHANNELS_IMG, cfg.FEATURES_GEN).to(device)
    critic = Critic(cfg.CHANNELS_IMG, cfg.FEATURES_DISC).to(device)

    if args.checkpoint_path:
        opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
        opt_critic = optim.Adam(critic.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
        cp = torch.load(args.checkpoint_path)
        start_epoch, end_epoch, fixed_noise = load_checkpoint(cp, gen, critic, opt_gen, opt_critic)
        cfg.NUM_EPOCHS = end_epoch
    else:
        print("=> Init default weights of models and fixed noise")
        # FIXME sometime (usually) when the weights is initialized from normal distribution can cause mode collapse
        # init_weights(gen)
        # init_weights(disc)
        # defining optimizers after init weights
        opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
        opt_critic = optim.Adam(critic.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
        start_epoch = 1
        end_epoch = cfg.NUM_EPOCHS
        fixed_noise = get_random_noise(cfg.BATCH_SIZE, cfg.Z_DIMENSION, device)

    if args.resume_id:
        metric_logger = MetricLogger(cfg.PROJECT_VERSION_NAME, resume_id=args.resume_id)
    else:
        metric_logger = MetricLogger(cfg.PROJECT_VERSION_NAME)

    # gradients metric
    wandb.watch(gen)
    wandb.watch(critic)
    # model mode
    gen.train()
    critic.train()

    start_time = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        train_one_epoch(epoch, dataloader, gen, critic, opt_gen, opt_critic,
                        fixed_noise, device, metric_logger, num_samples=cfg.NUM_SAMPLES, freq=cfg.FREQ)
        if epoch == cfg.NUM_EPOCHS + 1:
            checkpoint(epoch, end_epoch, gen, critic, opt_gen, opt_critic, fixed_noise)
        elif epoch % cfg.SAVE_EACH_EPOCH == 0:
            checkpoint(epoch, end_epoch, gen, critic, opt_gen, opt_critic, fixed_noise)

    total_time = time.time() - start_time
    print(f"=> Training time:{total_time}")
