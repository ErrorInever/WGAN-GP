import unittest
from model import Critic, Generator


class TestModels(unittest.TestCase):

	def setUp(self):
		self.N = 8
		self.in_channels = 3
		self.H = 64
		self.W = 64
		self.noise_dim = 100
		self.x = torch.randn((N, in_channels, H, W))
		self.z = torch.randn((N, noise_dim, 1, 1))


	def test_shape_critic(self):
		disc = Critic(self.in_channels, 8)
		self.assertEqual(disc(self.x).shape == (self.N, 1, 1, 1))

	def test_shape_generator(self):
		gen = Generator(self.noise_dim, self.in_channels, 8)
		self.assertEqual(gen(self.z).shape == (self.N, self.in_channels, self.H, self.W))