import time
import os
import numpy as np
import losswise
import torchvision
import errno
import torch
import pickle
from matplotlib import pyplot as plt
from IPython import display
from config import cfg




class Metric_logger:
    """Metric class"""
    def __init__(self, project_name, losswise_api_key, show_accuracy=False):
        self.project_name = project_name
        self.losswise_api_key = losswise_api_key
        self.show_acc = show_accuracy
        self.data_subdir = f"{os.path.join(cfg.OUT_DIR, self.project_name)}/imgdata"

        self.loss = {'D':[], 'G':[]}
        self.acc = {'Dr':[], 'Df':[]}

        if self.losswise_api_key:
            print("init losswise api")
            losswise.set_api_key(self.losswise_api_key)
            self.session = losswise.Session(
                tag=project_name,
                max_iter=cfg.NUM_EPOCHS,
                track_git=False
            )
            self.losswise_loss = self.session.graph('loss', kind='min')
            if self.show_acc:
                self.graph_acc = self.session.graph('accuracy', kind='max')
            print("Done")



    @staticmethod
    def display_status(epoch, num_epochs, batch_idx, num_batches, dis_loss, gen_loss, acc_real=None, acc_fake=None):
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
        if dis_loss and isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if gen_loss and isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if acc_real and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if acc_fake and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        #epoch = f"Batch[{batch_idx}/{num_batches}] Epoch[{epoch}/{num_epochs}]"
        #loss = f"Discriminator loss:{dis_loss=:.4f} Generator loss:{gen_loss=:.4f}"

        print('Batch Num: [{}/{}], Epoch: [{}/{}]'.format(batch_idx, num_batches, epoch, num_epochs))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dis_loss, gen_loss))
        if acc_real and acc_fake:
            #acc = f"D(x):{acc_real=:.4f} D(G(z)):{acc_fake=:.4f}"
            print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(acc_real, acc_fake))


    def log(self, epoch, batch_idx, num_batches, dis_loss, gen_loss, acc_real=None, acc_fake=None):
        """
        Logging values
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        :param dis_loss: ``torch.autograd.Variable``, critical loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, D(x) predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, D(G(z)) paramredicted on fake data
        """
        if dis_loss and isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if gen_loss and isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if acc_real and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if acc_fake and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        step = Metric_logger._step(epoch, batch_idx, num_batches)

        self.loss['D'].append(dis_loss)
        self.loss['G'].append(gen_loss)
        if self.show_acc:
            self.acc['Dr'].append(acc_real) # acc on real data
            self.acc['Df'].append(acc_fake) # acc on fake data
            self.graph_acc.append(step, {'D(x)': acc_real, 'D(G(z))': acc_fake})

        if self.losswise_api_key:
            self.losswise_loss.append(step, {'Critic': dis_loss, 'Generator': gen_loss})


    @staticmethod
    def _step(epoch, batch_idx, num_batches):
        return epoch * num_batches + batch_idx


    def log_image(self, images, num_samples, epoch, batch_idx, num_batches, normalize=True):
        """
        Create image grid and save it
        :param images: ``Torch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, number of batches
        :param normalize: if True normalize images
        """
        images = images[:num_samples, ...]
        horizontal_grid = torchvision.utils.make_grid(images, normalize=normalize, scale_each=True)
        nrows = int(np.sqrt(num_samples))
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        step = Metric_logger._step(epoch, batch_idx, num_batches)
        img_name = f'{self.project_name}/images{step}'
        self.save_torch_images(horizontal_grid, grid, epoch, batch_idx)


    def save_torch_images(self, horizontal_grid, grid, epoch, batch_idx, plot_horizontal=True, figsize=(16, 16)):
        """
        Display and save image grid
        :param horizontal_grid: ``ndarray``, horizontal grid image
        :param grid: ``ndarray``, grid image
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param plot_horizontal: if True plot horizontal grid image
        :param figsize: ``tuple``, figure size
        """
        out_dir = self.data_subdir
        fig = plt.figure(figsize=figsize)
        plt.imshow(np.moveaxis(horizontal_grid.detach().cpu().numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        Metric_logger._save_images(fig, epoch, batch_idx, out_dir)
        plt.close()
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        Metric_logger._save_images(fig, epoch, batch_idx, out_dir)
        plt.close()


    def _close_losswise_session(self):
        self.session.done()


    def save_local_metrics(self):
        with open(f'{cfg.OUT_DIR}/loss_g.dt', 'wb') as fp:
            pickle.dump(self.loss['G'], fp)
        with open(f'{cfg.OUT_DIR}/loss_d.dt', 'wb') as fp:
            pickle.dump(self.loss['D'], fp)
        if self.show_acc:
            with open(f'{cfg.OUT_DIR}/acc_real.dt', 'wb') as fp:
                pickle.dump(self.acc['Dr'], fp)
            with open(f'{cfg.OUT_DIR}/acc_fake.dt', 'wb') as fp:
                pickle.dump(self.acc['Df'], fp)


    @staticmethod
    def _save_images(fig, epoch, batch_idx, out_dir, comment=''):
        Metric_logger._make_dir(out_dir)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir, comment, epoch, batch_idx))

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
