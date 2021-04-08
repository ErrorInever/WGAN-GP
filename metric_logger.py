import losswise
import torch
import os
import errno
import pickle
import numpy as np
import torchvision
import logging
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from IPython import display
from config.conf import cfg

logger = logging.getLogger(__name__)


class MetricLogger:
    """Helper metric logger class"""
    def __init__(self, model_name, dataset_name, losswise_api_key, tensorboard=False):
        """
        :param model_name: ``str``, model name
        :param dataset_name: ``str``, dataset name
        :param losswise_api_key: ``str``, losswise API key
        :param tensorboard: if True uses tensorboard
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.ls_api_key = losswise_api_key
        self.tensorboard = tensorboard

        self.data_subdir = "{}/{}".format(model_name, dataset_name)
        self.comment = "{}/{}".format(model_name, dataset_name)

        self.step = []
        self.loss_g = []
        self.loss_d = []

        if self.ls_api_key:
            logger.info('Init losswise session')
            losswise.set_api_key(self.ls_api_key)
            self.session = losswise.Session(
                    tag=model_name,
                    max_iter=cfg.NUM_EPOCHS,
                    track_git=False
                    )
            self.graph_loss = self.session.graph('loss', kind='min')

        if self.tensorboard:
            self.tf_logger = SummaryWriter(comment=self.comment)

    def log(self, D_loss, G_loss, epoch, n_batch, num_batches):
        """
        Logging training values
        :param D_loss: ``torch.autograd.Variable``, discriminator loss
        :param G_loss: ``torch.autograd.Variable``, generator loss
        :param epoch: ``int``, current epoch
        :param n_batch: ``int``, current batch
        :param num_batches: ``int``, number of batches
        """
        if isinstance(D_loss, torch.autograd.Variable):
            D_loss = D_loss.item()
        if isinstance(G_loss, torch.autograd.Variable):
            G_loss = G_loss.item()

        step = MetricLogger._step(epoch, n_batch, num_batches)

        self.loss_d.append(D_loss)
        self.loss_g.append(G_loss)
        self.step.append(step)

        if self.ls_api_key:
            self.graph_loss.append(step, {'Discriminator': D_loss, 'Generator': G_loss})

        if self.tensorboard:
            self.tf_logger.add_scalar('loss/dis', D_loss, step)
            self.tf_logger.add_scalar('loss/gen', G_loss, step)

    def log_image(self, images, num_samples, epoch, n_batch, num_batches, normalize=True):
        """
        Logging images
        :param images: ``Torch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param epoch: ``int``, current epoch
        :param n_batch: ``int``, current batch
        :param num_batches: ``int``, number of batches
        :param normalize: if True normalize images
        """
        images = images[:num_samples, ...]
        horizontal_grid = torchvision.utils.make_grid(images, normalize=normalize, scale_each=True)
        nrows = int(np.sqrt(num_samples))
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        step = MetricLogger._step(epoch, n_batch, num_batches)
        img_name = '{}/images{}'.format(self.comment, '')

        self.tf_logger.add_image(img_name, horizontal_grid, step)
        self.save_torch_images(horizontal_grid, grid, epoch, n_batch)

    def save_torch_images(self, horizontal_grid, grid, epoch, n_batch, plot_horizontal=True, figsize=(16, 16)):
        """
        Plot and save grid images
        :param horizontal_grid: ``ndarray``, horizontal grid image
        :param grid: ``ndarray``, grid image
        :param epoch: ``int``, current epoch
        :param n_batch: ``int``, current batch
        :param plot_horizontal: if True plot horizontal grid image
        :param figsize: ``tuple``, figure size
        """
        out_dir = os.path.join(cfg.OUT_DIR, 'data/images{}'.format(self.data_subdir))
        MetricLogger._make_dir(out_dir)

        fig = plt.figure(figsize=figsize)
        plt.imshow(np.moveaxis(horizontal_grid.detach().cpu().numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        self._save_images(fig, epoch, n_batch, out_dir)
        plt.close()

        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        self._save_images(fig, epoch, n_batch, out_dir)
        plt.close()

    def _save_images(self, fig, epoch, n_batch, out_dir, comment=''):
        MetricLogger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, n_batch))

    def close(self):
        self.session.done()
        self.tf_logger.close()

    def save_models(self, generator, discriminator, epoch):
        out_dir = './data/models/{}'.format(self.data_subdir)
        MetricLogger._make_dir(out_dir)
        torch.save(generator.state_dict(),
                   '{}/G_epoch_{}'.format(out_dir, epoch))
        torch.save(discriminator.state_dict(),
                   '{}/D_epoch_{}'.format(out_dir, epoch))

    @staticmethod
    def checkpoint(epoch, model, optimizer):
        out_dir = os.path.join(cfg.OUT_DIR, './data/checkpoints')
        MetricLogger._make_dir(out_dir)
        save_name = os.path.join(out_dir, '{}_{}.pth'.format(model.__class__.__name__, epoch))
        MetricLogger._save_checkpoint({
            'start_epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name
        )
        print('Save model {}'.format(save_name))

    @staticmethod
    def _save_checkpoint(state, filename):
        torch.save(state, filename)

    def dump_metrics(self):
        with open('data/loss_g.txt', 'wb') as fp:
            pickle.dump(self.loss_g, fp)
        with open('data/loss_d.txt', 'wb') as fp:
            pickle.dump(self.loss_d, fp)

    @staticmethod
    def load_metrics():
        with open('data/loss_g.txt', 'rb') as fp:
            loss_g = pickle.load(fp)
        with open('data/loss_d.txt', 'rb') as fp:
            loss_d = pickle.load(fp)
        return loss_g, loss_d

    def plot_metrics(self):
        x = np.arange(0, len(self.step))
        loss_g = np.asarray(self.loss_g)
        loss_d = np.asarray(self.loss_d)

        fig = plt.figure(figsize=(20, 12))
        plt.style.use('fast')
        plt.grid()
        plt.title('Loss')
        plt.xlabel('iter')
        plt.ylabel('count')
        plt.plot(x, loss_g, label='Generator loss')
        plt.plot(x, loss_d, label='Discriminator loss')
        plt.legend()
        fig.savefig('loss.png')
        plt.close()

    @staticmethod
    def display_status(epoch, num_epochs, n_batch, num_batches, dis_loss, gen_loss):
        """
        Display training progress
        :param epoch: ``int``, current epoch
        :param num_epochs: ``int``, numbers epoch
        :param n_batch: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        :param dis_loss: ``torch.autograd.Variable``, discriminator loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()

        print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch, num_epochs, n_batch, num_batches))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dis_loss, gen_loss))

    @staticmethod
    def _step(epoch, n_batch, num_batches):
        return epoch * num_batches + n_batch

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
