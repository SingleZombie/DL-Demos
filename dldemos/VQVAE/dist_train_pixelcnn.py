import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from dldemos.VQVAE.configs import get_cfg
from dldemos.VQVAE.dataset import get_dataloader
from dldemos.VQVAE.model import VQVAE
from dldemos.VQVAE.pixelcnn_model import PixelCNNWithEmbedding

USE_LMDB = True


def reduce_sum(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def train_generative_model(vqvae: VQVAE,
                           model,
                           img_shape=None,
                           device='cuda',
                           ckpt_path='dldemos/VQVAE/gen_model.pth',
                           dataset_type='MNIST',
                           batch_size=64,
                           n_epochs=50):
    print('batch size:', batch_size)
    dataloader, sampler = get_dataloader(dataset_type,
                                         batch_size,
                                         img_shape=img_shape,
                                         dist_train=True,
                                         use_lmdb=USE_LMDB)
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0
        sampler.set_epoch(e)
        for x in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)

            predict_x = model(x)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = reduce_sum(loss)
            total_loss += loss * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        if device == 0:
            torch.save(model.module.state_dict(), ckpt_path)
            print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
        dist.barrier()

    print('Done')


if __name__ == '__main__':
    dist.init_process_group('nccl')

    os.makedirs('work_dirs', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=0)
    args = parser.parse_args()
    cfg = get_cfg(args.c)

    img_shape = cfg['img_shape']
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding'])
    gen_model = PixelCNNWithEmbedding(cfg['pixelcnn_n_blocks'],
                                      cfg['pixelcnn_dim'],
                                      cfg['pixelcnn_linear_dim'], True,
                                      cfg['n_embedding'])

    # 3. Train Generative model (Gated PixelCNN in our project)
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
    vqvae.to(device)
    gen_model.to(device)
    gen_model = DistributedDataParallel(gen_model, device_ids=[device])

    # Optional: resume
    # map_location = {'cuda:0': f'cuda:{device}'}
    # state_dict = torch.load(cfg['gen_model_path'], map_location=map_location)
    # gen_model.module.load_state_dict(state_dict)

    train_generative_model(vqvae,
                           gen_model,
                           img_shape=(img_shape[1], img_shape[2]),
                           device=device,
                           ckpt_path=cfg['gen_model_path'],
                           dataset_type=cfg['dataset_type'],
                           batch_size=cfg['batch_size_2'],
                           n_epochs=cfg['n_epochs_2'])

    dist.destroy_process_group()

# torchrun --nproc_per_node=4 dldemos/VQVAE/dist_train_pixelcnn.py -c 1
