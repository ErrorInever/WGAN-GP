import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Critic, Generator, init_weights
from config import cfg


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
loader = DataLoader(
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

# draw results
fixed_noise = torch.rand(32, cfg.Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()

# main loop

for epoch in range(cfg.NUM_EPOCHS):
	for batch_idx, (real, _) in enumerate(loader):
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

		# print to tensorboard
		if batch_idx % 100 == 0 and batch_idx > 0:
			print(
				f"Epoch [{epoch}/{cfg.NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
				Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
			)

			with torch.no_grad():
				fake = gen(fixed_noise)
				# take out (up to) 32 examples
				img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
				img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
				writer_real.add_image("Real", img_grid_real, global_step=step)
				writer_fake.add_image("Fake", img_grid_fake, global_step=step)

			step += 1
