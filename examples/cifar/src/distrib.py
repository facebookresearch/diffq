# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from submitit import JobEnvironment
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel.distributed import DistributedDataParallel

logger = logging.getLogger(__name__)
rank = 0
local_rank = 0
world_size = 1


def init_rank(args):
    global local_rank, rank, world_size
    try:
        job_env = JobEnvironment()
    except RuntimeError:
        if args.rank is not None:
            rank = args.rank
            local_rank = args.rank % 8
            world_size = args.world_size
        return
    rank = job_env.global_rank
    local_rank = job_env.local_rank
    world_size = job_env.num_tasks


def init(args, rendezvous_file):
    init_rank(args)
    if world_size == 1:
        return
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='file://' + os.path.abspath(rendezvous_file),
        world_size=world_size,
        rank=rank)
    logger.debug("Distributed rendezvous went well, rank %d/%d", rank, world_size)


def average(metrics, count=1.):
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()


def wrap(model):
    if world_size == 1:
        return model
    else:
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())


def barrier():
    if world_size > 1:
        torch.distributed.barrier()


def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    """
    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.
    """
    if world_size == 1:
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
        return klass(dataset, *args, shuffle=shuffle, **kwargs)
