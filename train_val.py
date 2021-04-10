import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
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
gen = Generator(cfg.Z_DIM, cfg.CHANNELS_IMG, cfg.FEATURES_GEN).to(device)
critic = Critic(cfg.CHANNELS_IMG, cfg.FEATURES_CRITIC).to(device)
# init weights
init_weights(gen)
init_weights(critic)
# init optimizers
opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
critic.train()

freq = 100
num_sumples = 16
static_noise = torch.randn(num_sumples, cfg.Z_DIM, 1, 1, device=device)
metric_logger = MetricLogger('WGAN-GP', 'MNIST', losswise_api_key=args.api_key, tensorboard=True)

# main loop

for epoch in range(cfg.NUM_EPOCHS):
	for n_batch, (real, _) in enumerate(dataloader):
		real = real.to(device)
		cur_batch_size = real.shape[0]

		# Train critic: max E[critic(real) - E[critic(fake)]] 
		# i.e. minimizing the negative of that
		for _ in range(cfg.CRITIC_ITERATIONS):
			noise = torch.randn(cur_batch_size, cfg.Z_DIM, 1, 1).to(device)
			fake = gen(noise)
			critic_real = critic(real).reshape(-1)
			critic_fake = critic(fake).reshape(-1)
			gp = gradient_penalty(critic, real, fake, device=device)
			loss_critic = (-1 * (torch.mean(critic_real) - torch.mean(critic_fake)) 
				+ cfg.LAMBDA_GRADIENT_PENALTY * gp)
			critic.zero_grad()
			loss_critic.backward(retain_graph=True)
			opt_critic.step()


		# Train generator: max E[critic(gen_fake)] <--> min -E(critic(gen_fake))
		gen_fake = critic(fake).reshape(-1)
		loss_gen = -1 * torch.mean(gen_fake)
		gen.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		if n_batch % freq == 0:
			with torch.no_grad():
				metric_logger.log(loss_critic, loss_gen, epoch, n_batch, len(dataloader))
				static_fake_data = gen(static_noise)
				metric_logger.log_image(static_fake_data, num_sumples, epoch, n_batch, len(dataloader))
				metric_logger.display_status(epoch, cfg.NUM_EPOCHS, n_batch, len(dataloader), 
					loss_critic, loss_gen)
