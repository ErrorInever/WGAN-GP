import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Critic, Generator, init_weights
from config import cfg
from metric_logger import MetricLogger 




def parse_args():
    parser = argparse.ArgumentParser(description='WGAN-GP')
    parser.add_argument('--api_key', dest='api_key', help='losswise api key', default=None, type=str)
    return parser.parse_args()


args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = transforms.Compose(
	[
		transforms.Resize(cfg.IMAGE_SIZE),
		transforms.ToTensor(),
		transforms.Normalize(
			[0.5 for _ in range(cfg.CHANNELS_IMG)], [0.5 for _ in range(cfg.CHANNELS_IMG)]),
	]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
dataloader = DataLoader(
	dataset,
	batch_size=cfg.BATCH_SIZE,
	shuffle=True,
)

# init models 
gen = Generator(cfg.LATENT_Z_DIMENSION, cfg.CHANNELS_IMG, cfg.FEATURES_GEN).to(device)
critic = Critic(cfg.CHANNELS_IMG, cfg.FEATURES_CRITIC).to(device)
# init weights of models
init_weights(gen)
init_weights(critic)
# init optimizers
opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
critic.train()

num_sumples = 16
static_noise = torch.randn(num_sumples, cfg.LATENT_Z_DIMENSION, 1, 1, device=device)
#metric_logger = MetricLogger('WGAN-GP', 'MNIST', losswise_api_key=args.api_key, tensorboard=True)

# main loop
metric_logger = MetricLogger('WGAN', args.api_key)

start_time = time.time()
for epoch in range(cfg.NUM_EPOCHS):
	train_one_epoch(epoch, dataloader, gen, critic, opt_gen, opt_critic, static_noise, 
		device, metric_logger, freq=100)


total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print(f"Training time {total_time_str}")