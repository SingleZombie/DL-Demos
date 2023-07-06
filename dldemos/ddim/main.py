import os
import time

import cv2
import einops
import torch
import torch.nn as nn

from dldemos.ddim.configs import configs
from dldemos.ddim.dataset import get_dataloader
from dldemos.ddim.ddim import DDIM
from dldemos.ddim.ddpm import DDPM
from dldemos.ddim.network import UNet


def train(ddpm: DDPM,
          net,
          dataset_type,
          resolution=None,
          batch_size=512,
          n_epochs=50,
          device='cuda',
          ckpt_path='dldemos/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(dataset_type,
                                batch_size,
                                resolution=resolution)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 2e-4)

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
    if img_shape[1] >= 256:
        max_batch_size = 16
    elif img_shape[1] >= 128:
        max_batch_size = 64
    else:
        max_batch_size = 256

    net = net.to(device)
    net = net.eval()

    index = 0
    with torch.no_grad():
        while n_sample > 0:
            if n_sample >= max_batch_size:
                batch_size = max_batch_size
            else:
                batch_size = n_sample
            n_sample -= batch_size
            shape = (batch_size, *img_shape)
            imgs = ddpm.sample_backward(shape,
                                        net,
                                        device=device,
                                        simple_var=simple_var,
                                        **kwargs).detach().cpu()
            imgs = (imgs + 1) / 2 * 255
            imgs = imgs.clamp(0, 255).to(torch.uint8)

            img_list = einops.rearrange(imgs, 'n c h w -> n h w c').numpy()
            output_dir = os.path.splitext(output_path)[0]
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(img_list):
                if to_bgr:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'{output_dir}/{i+index}.jpg', img)

            # First iteration
            if index == 0:
                imgs = einops.rearrange(imgs,
                                        '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                        b1=int(batch_size**0.5))
                imgs = imgs.numpy()
                if to_bgr:
                    imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, imgs)

            index += batch_size


if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)

    # 0 for MNIST. See configs.py
    config_id = 0
    cfg = configs[config_id]
    n_steps = 1000
    device = 'cuda'
    model_path = cfg['model_path']
    img_shape = cfg['img_shape']
    to_bgr = False if cfg['dataset_type'] == 'MNIST' else True

    net = UNet(n_steps, img_shape, cfg['channels'], cfg['pe_dim'],
               cfg.get('with_attn', False), cfg.get('norm_type', 'ln'))
    ddpm = DDPM(device, n_steps)

    train(ddpm,
          net,
          cfg['dataset_type'],
          resolution=(img_shape[1], img_shape[2]),
          batch_size=cfg['batch_size'],
          n_epochs=cfg['n_epochs'],
          device=device,
          ckpt_path=model_path)

    net.load_state_dict(torch.load(model_path))
    ddim = DDIM(device, n_steps)
    sample_imgs(ddpm,
                net,
                'work_dirs/diffusion_ddpm_sigma_hat.jpg',
                img_shape,
                device=device,
                to_bgr=to_bgr)
    sample_imgs(ddim,
                net,
                'work_dirs/diffusion_ddpm_eta_0.jpg',
                img_shape,
                device=device,
                to_bgr=to_bgr,
                ddim_step=1000,
                simple_var=False,
                eta=0)

    sample_imgs(ddim,
                net,
                'work_dirs/diffusion_ddim_sigma_hat.jpg',
                img_shape,
                device=device,
                simple_var=True,
                to_bgr=to_bgr)
    sample_imgs(ddim,
                net,
                'work_dirs/diffusion_ddim_eta_1.jpg',
                img_shape,
                device=device,
                simple_var=False,
                eta=1,
                to_bgr=to_bgr)
    sample_imgs(ddim,
                net,
                'work_dirs/diffusion_ddim_eta_0.jpg',
                img_shape,
                device=device,
                simple_var=False,
                eta=0,
                to_bgr=to_bgr)
