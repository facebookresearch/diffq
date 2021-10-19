#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import socket

import hydra

logger = logging.getLogger(__name__)


def run(args):
    from src import distrib
    from src import data
    from src import solver as slv
    from src.mobilenet import MobileNet
    from src.resnet import ResNet18
    from src.wide_resnet import Wide_ResNet

    import torch
    from diffq import DiffQuantizer, UniformQuantizer, LSQ

    logger.info("Running on host %s", socket.gethostname())
    distrib.init(args, args.rendezvous_file)

    # validate distributed training
    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    # setup data loaders
    trainset, testset, num_classes = data.get_loader(args, model_name=args.model.lower())
    tr_loader = distrib.loader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    tt_loader = distrib.loader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
    data = {"tr": tr_loader, "tt": tt_loader}

    # build the model
    if args.model.lower() == 'resnet':
        model = ResNet18(num_classes=num_classes)
    elif args.model.lower() == 'w_resnet':
        # WideResNet params
        depth = 28
        widen_factor = 10
        do = 0.3
        model = Wide_ResNet(depth=depth, widen_factor=widen_factor,
                            dropout_rate=do, num_classes=num_classes)
    elif args.model.lower() == 'mobilenet':
        model = MobileNet(num_classes=num_classes)
    else:
        print('Arch not supported.')
        os._exit(1)

    logger.debug(model)
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2**20
    logger.info('Size: %.1f MB', model_size)

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.bechmark = True

    params = model.parameters()
    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, args.beta2))
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr,
                                      betas=(0.9, args.beta2), weight_decay=args.w_decay)
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            params, lr=args.lr, weight_decay=args.w_decay,
            momentum=args.momentum, alpha=args.alpha)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.w_decay, nesterov=args.nestrov)
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    if args.quant.penalty:
        quantizer = DiffQuantizer(
            model, group_size=args.quant.group_size,
            min_size=args.quant.min_size,
            min_bits=args.quant.min_bits,
            init_bits=args.quant.init_bits,
            max_bits=args.quant.max_bits,
            exclude=args.quant.exclude)
        if args.quant.adam:
            quantizer.opt = torch.optim.Adam([{"params": []}])
            quantizer.setup_optimizer(quantizer.opt, lr=args.quant.lr)
        else:
            quantizer.setup_optimizer(optimizer, lr=args.quant.lr)
    elif args.quant.lsq:
        quantizer = LSQ(
            model, bits=args.quant.bits, min_size=args.quant.min_size,
            exclude=args.quant.exclude)
        quantizer.setup_optimizer(optimizer)
    elif args.quant.bits:
        quantizer = UniformQuantizer(
            model, min_size=args.quant.min_size,
            bits=args.quant.bits, qat=args.quant.qat, exclude=args.quant.exclude)
    else:
        quantizer = None
    criterion = torch.nn.CrossEntropyLoss()

    # Construct Solver
    solver = slv.Solver(data, model, criterion, optimizer, quantizer, args, model_size)
    solver.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("diffq_cifar").setLevel(logging.DEBUG)

    # Updating paths in config
    if args.continue_from:
        args.continue_from = os.path.join(
            os.getcwd(), "..", args.continue_from, args.checkpoint_file)
    args.db.root = hydra.utils.to_absolute_path(args.db.root + '/' + args.db.name)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)


@hydra.main(config_path="conf", config_name='config.yaml')
def main(args):
    try:
        if args.ddp and args.rank is None:
            from src.executor import start_ddp_workers
            start_ddp_workers(args)
            return
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        os._exit(1)  # a bit dangerous for the logs, but Hydra intercepts exit code
        # fixed in beta but I could not get the beta to work


if __name__ == "__main__":
    main()
