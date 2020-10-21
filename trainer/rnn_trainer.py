import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseMultiTrainer
from utils import MetricTracker


class Trainer(BaseMultiTrainer):
    """
    Trainer class
    """
    def __init__(self, model_idx, models, criterion, metric_ftns, optimizers, config, padding_idx, data_loader,
                 init_token, valid_data_loader=None, lr_schedulers=None, len_epoch=2):
        super().__init__(models, criterion, metric_ftns, optimizers, config)
        self.model_idx = model_idx
        self.config = config
        self.padding_idx = padding_idx
        self.init_token = init_token
        self.data_loader = data_loader
        self.batch_size = data_loader.batch_size
        self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_schedulers = lr_schedulers
        self.clip = self.config['trainer']['clip']
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        for idx in range(len(self.models)):
            self.models[idx].train()

        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            talk, response = data.talk, data.response
            talk_seq, talk_seq_len = talk[0].to(self.device), talk[1].to(self.device)
            response_seq, response_seq_len = response[0].to(self.device), response[1].to(self.device)
            # mask is TARGET mask
            mask = (response_seq != self.padding_idx)
            mask = mask.to(self.device)
            for idx in range(len(self.optimizers)):
                self.optimizers[idx].zero_grad()
            encoder_outputs, encoder_hidden = self.models[self.model_idx['encoder']](talk_seq, talk_seq_len)

            decoder_input = torch.LongTensor([[self.init_token for _ in range(self.batch_size)]])
            decoder_input = decoder_input.to(self.device)

            # TODO: test different init hidden
            # encoder_h = encoder_hidden.view(
            #     self.models[self.model_idx['encoder']].n_layers,
            #     2, -1, self.models[self.model_idx['encoder']].hidden_size
            # )[-1:, :1]
            # decoder_hidden = encoder_hidden[-1].expand(
            #     self.models[self.model_idx['decoder']].n_layers, -1, -1
            # )
            decoder_hidden = encoder_hidden[-self.models[self.model_idx['decoder']].n_layers:]

            loss = torch.tensor(0)
            losses = []
            n_totals = torch.tensor(0)
            for t in range(self.data_loader.sent_len):
                decoder_output, decoder_hidden = self.models[self.model_idx['decoder']](
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_input = response_seq[t:t+1]
                mask_loss, n_total = self.criterion(decoder_output, response_seq[t], mask[t])
                loss += mask_loss
                losses += [mask_loss.item() * n_total]
                n_totals += n_total

            loss.backward()
            for idx in range(len(self.optimizers)):
                torch.nn.utils.clip_grad_norm_(self.optimizers[idx], self.clip)
                self.optimizers[idx].step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_schedulers is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
