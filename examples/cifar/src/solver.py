# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path
import os
import time

import torch
from . import distrib
from .utils import bold, copy_state, pull_metric, LogProgress


logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, model, criterion, optimizer, quantizer, args, model_size):
        self.tr_loader = data['tr']
        self.tt_loader = data['tt']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.quantizer = quantizer
        self.penalty = args.quant.penalty
        self.model_size = model_size
        self.compressed_model_size = model_size

        if args.lr_sched == 'step':
            from torch.optim.lr_scheduler import StepLR
            sched = StepLR(self.optimizer, step_size=args.step.step_size, gamma=args.step.gamma)
        elif args.lr_sched == 'multistep':
            from torch.optim.lr_scheduler import MultiStepLR
            sched = MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=args.multistep.gamma)
        elif args.lr_sched == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            sched = ReduceLROnPlateau(
                self.optimizer, factor=args.plateau.factor, patience=args.plateau.patience)
        elif args.lr_sched == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            sched = CosineAnnealingLR(
                self.optimizer, T_max=args.cosine.T_max, eta_min=args.cosine.min_lr)
        else:
            sched = None
        self.sched = sched

        # Training config
        self.device = args.device
        self.epochs = args.epochs
        self.max_norm = args.max_norm

        # Checkpoints
        self.continue_from = args.continue_from
        self.checkpoint = Path(
            args.checkpoint_file) if args.checkpoint else None
        if self.checkpoint:
            logger.debug("Checkpoint will be saved to %s", self.checkpoint.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        # keep track of losses
        self.history = []

        # logging
        self.num_prints = args.num_prints

        if args.mixed:
            self.scaler = torch.cuda.amp.GradScaler()

        # for seperation tests
        self.args = args
        self._reset()

    def _serialize(self, path):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        if self.quantizer and hasattr(self.quantizer, 'opt'):
            package['quant_opt'] = self.quantizer.opt.state_dict()
        if self.args.mixed:
            package['scaler'] = self.scaler.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint) + ".tmp"

        torch.save(package, tmp_path)
        os.rename(tmp_path, path)

    def _reset(self):
        load_from = None
        # Reset
        if self.checkpoint and self.checkpoint.exists() and not self.restart:
            load_from = self.checkpoint
        elif self.continue_from:
            load_from = self.continue_from

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_from == self.continue_from and self.args.continue_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['state'])
            if 'optimizer' in package and not self.args.continue_best:
                # Ignore LR from checkpoint as we replay all scheduler steps
                package['optimizer']['param_groups'][0]["lr"] = self.args.lr
                self.optimizer.load_state_dict(package['optimizer'])
            if self.quantizer and hasattr(self.quantizer, 'opt'):
                self.quantizer.opt.load_state_dict(package['quant_opt'])
            if self.args.mixed:
                self.scaler.load_state_dict(package['scaler'])
            self.history = package['history']
            self.best_state = package['best_state']

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch}: {info}")
            if self.sched is not None:
                self.sched.step()

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss, train_acc = self._run_one_epoch(epoch)
            logger.info(bold(f'Train Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | train loss {train_loss:.5f} | '
                             f'train accuracy {train_acc:.2f}'))
            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout & Diffq

            with torch.no_grad():
                valid_loss, valid_acc = self._run_one_epoch(epoch, cross_valid=True)
            logger.info(bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s | valid Loss {valid_loss:.5f} | '
                             f'valid accuracy {valid_acc:.2f}'))
            if self.quantizer:
                logger.info(bold(f'True model size is {self.quantizer.true_model_size():.2f} MB'))
                self.model_size = self.quantizer.true_model_size()
                self.compressed_model_size = self.quantizer.compressed_model_size()

            # learning rate scheduling
            if self.sched:
                if self.args.lr_sched == 'plateau':
                    self.sched.step(valid_loss)
                else:
                    self.sched.step()
                new_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info(f'Learning rate adjusted: {new_lr:.5f}')

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'train_acc': train_acc,
                       'valid': valid_loss, 'valid_acc': valid_acc, 'best': best_loss,
                       'compressed_model_size': self.compressed_model_size,
                       'model_size': self.model_size}
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize(self.checkpoint)
                    logger.debug("Checkpoint saved to %s", self.checkpoint.resolve())

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        total = 0
        correct = 0
        data_loader = self.tr_loader if not cross_valid else self.tt_loader
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, (inputs, targets) in enumerate(logprog):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if not cross_valid:
                with torch.cuda.amp.autocast(bool(self.args.mixed)):
                    yhat = self.dmodel(inputs)
                    loss = self.criterion(yhat, targets)
                    model_size = self.quantizer.model_size() if self.quantizer else 0
                    if self.penalty > 0:
                        loss = loss + self.penalty * model_size
            else:
                # compute output
                yhat = self.dmodel(inputs)
                loss = self.criterion(yhat, targets)

            if not cross_valid:
                # optimize model in training mode
                self.optimizer.zero_grad()
                if self.quantizer and hasattr(self.quantizer, 'opt'):
                    self.quantizer.opt.zero_grad()

                if self.args.mixed:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    if self.quantizer and hasattr(self.quantizer, 'opt'):
                        self.scaler.unscale_(self.quantizer.opt)
                else:
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_norm)
                if self.args.mixed:
                    self.scaler.step(self.optimizer)
                else:
                    self.optimizer.step()
                if self.quantizer and hasattr(self.quantizer, 'opt'):
                    if self.args.mixed:
                        self.scaler.step(self.quantizer.opt)
                    else:
                        self.quantizer.opt.step()
                if self.args.mixed:
                    self.scaler.update()

            total_loss += loss.item()
            _, predicted = yhat.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            total_acc = 100. * (correct / total)
            if not cross_valid:
                logprog.update(
                    loss=format(total_loss / (i + 1), ".5f"),
                    accuracy=format(total_acc, ".5f"), MS=format(model_size, ".3f"))
            else:
                logprog.update(loss=format(total_loss / (i + 1), ".5f"),
                               accuracy=format(total_acc, ".5f"))
            # Just in case, clear some memory
            del loss
        return (distrib.average([total_loss / (i + 1)], i + 1)[0],
                distrib.average([total_acc], total)[0])
