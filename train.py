import torch
import time
from functions import gradient_penalty
from config import cfg

def train_one_epoch(epoch, dataloader, gen, critic, opt_gen, opt_critic,
	static_noise, device, metric_logger, num_samples, freq=100):
	"""
	Train models
	"""
	for batch_idx, (real_data, _) in enumerate(dataloader):
		real_data = real_data.to(device)
		cur_batch_size = real_data.shape[0]

		# Train critic: maximize E[critic(real) - E[critic(fake)]]
		# i.e. minimizing the negative of that
		for _ in range(cfg.CRITIC_ITERATIONS):
			noise = torch.randn(cur_batch_size, cfg.LATENT_Z_DIMENSION, 1, 1).to(device)
			fake_data = gen(noise)
			critic_real = critic(real_data).reshape(-1)
			critic_fake = critic(fake_data).reshape(-1)
			# gradient penalty
			grad_pen = gradient_penalty(critic, real_data, fake_data, device)
			# loss function
			loss_critic = (-1 * (torch.mean(critic_real) - torch.mean(critic_fake)) 
				+ cfg.LAMBDA * grad_pen)

			critic.zero_grad()
			loss_critic.backward(retain_graph=True)
			opt_critic.step()

		# Train generator: maximize E[critic(gen_fake)] <--> minimize -E(critic(gen_fake))
		gen_fake = critic(fake_data).reshape(-1)
		loss_gen = -1 * torch.mean(gen_fake)
		gen.zero_grad()
		loss_gen.backward()
		opt_gen.step()

		# Metrics
		if batch_idx % freq == 0 and batch_idx > 0:
			static_fake_data = gen(static_noise)
			metric_logger.log(epoch, batch_idx, len(dataloader), loss_critic, loss_gen)
			metric_logger.log_image(static_fake_data, num_samples, epoch, 
				batch_idx, len(dataloader))
			metric_logger.display_status(epoch, cfg.NUM_EPOCHS, batch_idx, 
				len(dataloader), loss_critic, loss_gen)
