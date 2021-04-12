import argparse
import torchvision
from matplotlib import pyplot as plt
from model import Generator
from utils import load_gen
from config import cfg


def parse_args():
	parser = argparse.ArgumentParser(description='WGAN-GP')
	parser.add_argument('--num_samples', dest='num_samples', help='digits count', default=None, type=str)
	parser.add_argument('--model_path', dest='model_path', help='path to weights of model', default=None, type=str)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args
	assert args.model_path, 'model_path not specified'

	if args.num_samples is None:
		num_samples = 32
	else:
		num_samples = args.num_samples

	device = "cuda" if torch.cuda.is_available() else "cpu"

	print("=> Load model weights")
	gen = Generator(cfg.LATENT_Z_DIMENSION, cfg.CHANNELS_IMG, cfg.FEATURES_GEN, 
	cfg.NUM_CLASSES, cfg.IMG_SIZE, cfg.GEN_EMBEDDING).to(device)
	load_gen(gen, args.model_path)

	noise = torch.randn(num_samples, 100, 1, 1).to(device)
	labels = torch.LongTensor(np.random.randint(0, 10, size=num_samples)).to(device)

	fake_data = gen(noise, labels)

	images = out
	horizontal_grid = torchvision.utils.make_grid(images, normalize=True, scale_each=True)
	nrows = int(np.sqrt(num_samples))
	grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=True, scale_each=True)
	fig.savefig(f'result_{num_samples}')
