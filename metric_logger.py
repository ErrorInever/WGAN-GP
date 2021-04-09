import time
import numpy as np


class Metric_logger:
    """Helper class"""
    def __init__(self, project_name):
        self.project_name = project_name


    @staticmethod
    def display_status(epoch, num_epochs, batch_idx, num_batches, dis_loss, gen_loss, acc_real, acc_fake):
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

        epoch = f"Batch[{batch_idx}/{num_batches}] Epoch[{epoch}/{num_epochs}]"
        loss = f"Discriminator loss:{dis_loss=:.4f} Generator loss:{gen_loss=:.4f}"

        print(epoch)
        print(loss)
        if acc_real and acc_fake:
            acc = f"D(x):{acc_real=:.4f} D(G(z)):{acc_fake=:.4f}"
            print(acc)


    def log(self, epoch, batch_idx, num_batches, dis_loss, gen_loss, acc_real, acc_fake):
        """Logging values"""
        if dis_loss and isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if gen_loss and isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if acc_real and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if acc_fake and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        step = Metric_logger._step(epoch, batch_idx, num_batches)

        # local metrics

        # losswise

        # tensorboard

    @staticmethod
    def _step(epoch, batch_idx, num_batches):
        return epoch * num_batches + batch_idx
