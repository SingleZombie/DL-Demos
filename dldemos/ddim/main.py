import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn

from dldemos.ddim.dataset import get_dataloader, get_img_shape
from dldemos.ddim.ddpm import DDPM
from dldemos.ddim.ddim import DDIM
from dldemos.ddim.network import UNet


def train(ddpm: DDPM,
          net,
          dataset_type,
          batch_size=512,
          n_epochs=50,
          device='cuda',
          ckpt_path='dldemos/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(dataset_type, batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

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
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample_imgs(ddpm,
                net,
                output_path,
                img_shape,
                n_sample=64,
                device='cuda',
                simple_var=True,
                to_bgr=False,
                **kwargs):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        shape = (n_sample, *img_shape)  # 1, 3, 28, 28 for MNIST
        imgs = ddpm.sample_backward(shape,
                                    net,
                                    device=device,
                                    simple_var=simple_var,
                                    **kwargs).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs,
                                '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)
        if to_bgr:
            imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, imgs)


mnist_cfg = {
    'dataset_type': 'MNIST',
    'model_path': 'dldemos/ddim/mnist.pth',
    'batch_size': 512,
    'n_epochs': 50,
    'channels': [10, 20, 40, 80],
    'pe_dim': 128
}
celebahq_cfg1 = {
    'dataset_type': 'CelebAHQ',
    'model_path': 'dldemos/ddim/celebahq.pth',
    'batch_size': 64,
    'n_epochs': 100,
    'channels': [32, 64, 128, 256],
    'pe_dim': 128
}
celebahq_cfg2 = {
    'dataset_type': 'CelebAHQ',
    'model_path': 'dldemos/ddim/celebahq2.pth',
    'batch_size': 64,
    'n_epochs': 100,
    'channels': [16, 32, 64, 128, 256],
    'pe_dim': 128
}
celebahq_cfg3 = {
    'dataset_type': 'CelebAHQ',
    'model_path': 'dldemos/ddim/celebahq3.pth',
    'batch_size': 32,
    'n_epochs': 100,
    'channels': [16, 32, 64, 128, 256],
    'pe_dim': 128,
    'with_attn': [False, False, False, True, True]
}
celebahq_cfg4 = {
    'dataset_type': 'CelebAHQ',
    'model_path': 'dldemos/ddim/celebahq4.pth',
    'batch_size': 64,
    'n_epochs': 100,
    'channels': [32, 64, 128, 256, 256],
    'pe_dim': 128
}
celebahq_cfg5 = {
    'dataset_type': 'CelebAHQ',
    'model_path': 'dldemos/ddim/celebahq5.pth',
    'batch_size': 64,
    'n_epochs': 100,
    'channels': [32, 64, 128, 256, 256],
    'pe_dim': 128,
    'with_attn': [False, False, False, True, True]
}
configs = [
    mnist_cfg, celebahq_cfg1, celebahq_cfg2, celebahq_cfg3, celebahq_cfg4,
    celebahq_cfg5
]

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    config_id = 5
    cfg = configs[config_id]
    n_steps = 1000
    device = 'cuda'
    model_path = cfg['model_path']
    img_shape = get_img_shape(cfg['dataset_type'])
    to_bgr = False if cfg['dataset_type'] == 'MNIST' else True

    net = UNet(n_steps, img_shape, cfg['channels'], cfg['pe_dim'],
               cfg.get('with_attn', False))
    ddpm = DDPM(device, n_steps)

    train(ddpm,
          net,
          cfg['dataset_type'],
          batch_size=cfg['batch_size'],
          n_epochs=cfg['n_epochs'],
          device=device,
          ckpt_path=model_path)

    net.load_state_dict(torch.load(model_path))
    # sample_imgs(ddpm,
    #             net,
    #             'work_dirs/diffusion_ddpm.jpg',
    #             img_shape,
    #             device=device,
    #             to_bgr=to_bgr)

    # ddim = DDIM(device, n_steps)
    # sample_imgs(ddim,
    #             net,
    #             'work_dirs/diffusion_ddim_sigma_hat.jpg',
    #             img_shape,
    #             device=device,
    #             simple_var=True,
    #             to_bgr=to_bgr)
    # sample_imgs(ddim,
    #             net,
    #             'work_dirs/diffusion_ddim_eta_1.jpg',
    #             img_shape,
    #             device=device,
    #             simple_var=False,
    #             eta=1,
    #             to_bgr=to_bgr)
    # sample_imgs(ddim,
    #             net,
    #             'work_dirs/diffusion_ddim_eta_0.jpg',
    #             img_shape,
    #             device=device,
    #             simple_var=False,
    #             eta=0,
    #             to_bgr=to_bgr)
