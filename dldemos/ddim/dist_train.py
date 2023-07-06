import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from dldemos.ddim.configs import configs
from dldemos.ddim.dataset import get_dataloader
from dldemos.ddim.ddpm import DDPM
from dldemos.ddim.network import UNet


def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def train(ddpm: DDPM,
          net,
          dataset_type,
          resolution=None,
          batch_size=512,
          n_epochs=50,
          scheduler_cfg=None,
          device='cuda',
          ckpt_path='dldemos/ddpm/model.pth'):

    n_steps = ddpm.n_steps
    dataloader, sampler = get_dataloader(dataset_type,
                                         batch_size,
                                         True,
                                         resolution=resolution)
    if device == 0:
        print('batch size: ', batch_size * dist.get_world_size())
        print('batch size per device: ', batch_size)

    net = net.to(device)
    loss_fn = nn.MSELoss()

    if scheduler_cfg is not None:
        optimizer = torch.optim.Adam(net.parameters(), scheduler_cfg['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, scheduler_cfg['milestones'], scheduler_cfg['gamma'])
    else:
        optimizer = torch.optim.Adam(net.parameters(), 2e-4)
        scheduler = None

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        sampler.set_epoch(e)
        for x in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = reduce_sum(loss)
            total_loss += loss.item() * current_batch_size
            if scheduler is not None:
                scheduler.step()
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        if device == 0:
            torch.save(net.module.state_dict(), ckpt_path)
            print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
        dist.barrier()

    if device == 0:
        print('Done')


if __name__ == '__main__':
    dist.init_process_group('nccl')

    os.makedirs('work_dirs', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=0)
    args = parser.parse_args()
    cfg = configs[args.c]

    n_steps = 1000
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    model_path = cfg['model_path']
    img_shape = cfg['img_shape']
    to_bgr = False if cfg['dataset_type'] == 'MNIST' else True

    net = UNet(n_steps, img_shape, cfg['channels'], cfg['pe_dim'],
               cfg.get('with_attn', False), cfg.get('norm_type', 'ln'))
    net.to(device)
    net = DistributedDataParallel(net, device_ids=[device])
    ddpm = DDPM(device, n_steps)

    # Optional: resume
    # map_location = {'cuda:0': f'cuda:{device}'}
    # resume_path = model_path
    # state_dict = torch.load(resume_path, map_location=map_location)
    # net.module.load_state_dict(state_dict)

    train(ddpm,
          net,
          cfg['dataset_type'],
          resolution=(img_shape[1], img_shape[2]),
          batch_size=cfg['batch_size'],
          n_epochs=cfg['n_epochs'],
          scheduler_cfg=cfg.get('scheduler_cfg', None),
          device=device,
          ckpt_path=model_path)

    dist.destroy_process_group()

# torchrun --nproc_per_node=8 dldemos/ddim/dist_train.py -c 1
