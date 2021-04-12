import torch
import torch.nn as nn


def init_weights(model):
	for m in model.modules():
		if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
			nn.init.normal_(m.weight.data, 0.0, 0.02)


class Generator(nn.Module):

	def __init__(self, z_dim, channels_img, features_g, num_classes, img_size, embed_size):
		super(Generator, self).__init__()
		self.img_size = img_size

		self.embed = nn.Embedding(num_classes, embed_size)
		self.model = nn.Sequential(
				# input dim: N x z_dim x 1 x 1
				self._block(z_dim + embed_size, features_g * 16, 4, 1, 0), # 4x4
				self._block(features_g * 16, features_g * 8, 4, 2, 1), # 8x8
				self._block(features_g * 8, features_g * 4, 4, 2, 1), # 16x16
				self._block(features_g * 4, features_g * 2, 4, 2, 1), # 32x32
				nn.ConvTranspose2d(
					features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
				),
				nn.Tanh()
			)

	def _block(self, in_channels, out_channels, kernel_size, stride, padding):
		return nn.Sequential(
				nn.ConvTranspose2d(
					in_channels,
					out_channels,
					kernel_size,
					stride,
					padding,
					bias=False
				),
				nn.BatchNorm2d(out_channels),
				nn.ReLU()
			)

	def forward(self, x, labels):
		# unput latent vector z: N x z_dim x 1 x 1
		embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) # add dimension
		x = torch.cat([x, embedding], dim=1) 
		return self.model(x)


class Critic(nn.Module):

	def __init__(self, channels_img, features_d, num_classes, img_size):
		super(Critic, self).__init__()
		self.img_size = img_size

		self.embed = nn.Embedding(num_classes, img_size * img_size)
		self.model = nn.Sequential(
			nn.Conv2d(channels_img + 1, features_d, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2),
			self._block(features_d, features_d * 2, 4, 2, 1),
			self._block(features_d * 2, features_d * 4, 4, 2, 1),
			self._block(features_d * 4, features_d * 8, 4, 2, 1),
			nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0)
		)


	def _block(self, in_channels, out_channels, kernel_size, stride, padding):
		return nn.Sequential(
				nn.Conv2d(
					in_channels,
					out_channels,
					kernel_size,
					stride,
					padding,
					bias=False
				),
				nn.InstanceNorm2d(out_channels, affine=True),
				nn.LeakyReLU(0.2)
			)


	def forward(self, x, labels):
		embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size)
		x = torch.cat([x, embedding], dim=1) # N x C x H(img_size) x w(img_size)
		return self.model(x)
