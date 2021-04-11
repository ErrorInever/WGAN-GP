import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import time
import random
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Critic, Generator, init_weights
from config import cfg
from metric_logger import Metric_logger 
from train import train_one_epoch
from utils import save_models, load_models


def parse_args():
	parser = argparse.ArgumentParser(description='WGAN-GP')
	parser.add_argument('--api_key', dest='api_key', help='losswise api key', default=None, type=str)
	parser.add_argument('--load_gen', dest='load_gen', help='loading generator', default=None, type=str)
	parser.add_argument('--load_crt', dest='load_crt', help='loading critic', default=None, type=str)
	return parser.parse_args()


def set_seed(val):
	"""
	Freezes random sequences
	:param val: ``int`` random value
	"""
	random.seed(val)
	np.random.seed(val)
	torch.manual_seed(val)
	torch.cuda.manual_seed(val)
	torch.backends.cudnn.deterministic = True


args = parse_args()
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=> Called with args {args.__dict__}")
print(f"=> Config params {cfg.__dict__}")
print(f"=> Run on device {device}")

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
print("=> Init models")
gen = Generator(cfg.LATENT_Z_DIMENSION, cfg.CHANNELS_IMG, cfg.FEATURES_GEN).to(device)
critic = Critic(cfg.CHANNELS_IMG, cfg.FEATURES_CRITIC).to(device)
# init weights of models
if args.load_gen and args.load_crt:
	load_models(gen, critic, args.load_gen, args.load_crt)
else:
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
metric_logger = Metric_logger('WGAN-GP', args.api_key)

start_time = time.time()
for epoch in range(cfg.NUM_EPOCHS):
	train_one_epoch(epoch, dataloader, gen, critic, opt_gen, opt_critic, static_noise, 
		device, metric_logger, num_sumples, freq=100)
	# save models
	if epoch % cfg.SAVE_EACH_EPOCH == 0:
		save_models(epoch, gen, critic)

total_time = time.time() - start_time
print(f"=> Training time {total_time}")
metric_logger.save_local_metrics()