import os
import numpy as np
import wandb
import torchvision
import errno
import torch
from matplotlib import pyplot as plt
from config import cfg


class MetricLogger:
    """Metric class"""
    def __init__(self, project_version_name, wab=True, show_accuracy=True, resume_id=False):
        """
        :param project_version_name: name of current version of project
        :param wab: good realtime metric, you can register free account in https://wandb.ai/
        :param show_accuracy: if True: show accuracy on real and fake data
        """
        self.project_version_name = project_version_name
        self.show_acc = show_accuracy
        self.data_subdir = f"{os.path.join(cfg.OUT_DIR, self.project_version_name)}/imgdata"

        if wab:
            if resume_id:
                wandb_id = resume_id
            else:
                wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id, project='DCGAN-Anime-Faces', name=project_version_name, resume=True)
            wandb.config.update({
                'train_images_count': cfg.DATASET_SIZE,
                'init_lr': cfg.LEARNING_RATE,
                'noise_z_size': cfg.Z_DIMENSION,
                'batch_size': cfg.BATCH_SIZE,
                'initialization_weights': 'Normal Distribution',
                'beta 1': 0.5,
                'beta 2': 0.999
            })

    def display_status(self, epoch, num_epochs, batch_idx, num_batches, dis_loss,
                       gen_loss, acc_real=None, acc_fake=None):
        """
        Display training progress
        :param epoch: ``int``, current epoch
        :param num_epochs: ``int``, numbers epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        :param dis_loss: ``torch.autograd.Variable``, discriminator loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, discriminator predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, discriminator predicted on fake data
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if self.show_acc and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if self.show_acc and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        print('Batch Num: [{}/{}], Epoch: [{}/{}]'.format(batch_idx, num_batches, epoch, num_epochs))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dis_loss, gen_loss))
        if acc_real and acc_fake:
            print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(acc_real, acc_fake))

    def log(self, dis_loss, gen_loss, acc_real=None, acc_fake=None):
        """
        Logging values
        :param dis_loss: ``torch.autograd.Variable``, critical loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, D(x) predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, D(G(z)) paramredicted on fake data
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if self.show_acc and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if self.show_acc and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        wandb.log({'d_loss': dis_loss, 'g_loss': gen_loss, 'D(x)': acc_real, 'D(G(z))': acc_fake})

    def log_image(self, images, num_samples, epoch, batch_idx, num_batches, normalize=True):
        """
        Create image grid and save it
        :param images: ``Tor    ch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param normalize: if True normalize images
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        """
        images = images[:num_samples, ...]
        nrows = int(np.sqrt(num_samples))
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        self.save_torch_images(grid, epoch, batch_idx, num_batches)
        wandb.log({'fixed_noise': [wandb.Image(np.moveaxis(grid.detach().cpu().numpy(), 0, -1))]})

    def save_torch_images(self, grid, epoch, batch_idx, num_batches):
        """
        Display and save image grid
        :param grid: ``ndarray``, grid image
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        """
        out_dir = self.data_subdir
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        MetricLogger._save_images(fig, out_dir, epoch, batch_idx, num_batches)
        plt.close()

    @staticmethod
    def _save_images(fig, out_dir, epoch, batch_idx, num_batches):
        """
        Saves image on drive
        :param fig: pls.figure object
        :param out_dir: path to output dir
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        """
        MetricLogger._make_dir(out_dir)
        image_name = f"epoch({str(epoch).zfill(len(str(cfg.NUM_EPOCHS)))})-" \
                     f"batch({str(batch_idx).zfill(len(str(num_batches)))}).jpg"
        fig.savefig('{}/{}'.format(out_dir, image_name))

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
