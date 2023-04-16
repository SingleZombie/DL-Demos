import torch
import torch.nn as nn
from dldemos.ddpm.dataset import get_dataloader, get_img_shape
from dldemos.ddpm.network import (build_network, convnet_medium_cfg,
                                  convnet_small_cfg, convnet_big_cfg,
                                  unet_1_cfg)
import time
import cv2
import numpy as np
import einops


class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self, img_shape, net, device):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net)
        return x

    def sample_backward_step(self, x, t, net):
        if t == 0:
            noise = 0
        else:
            noise = torch.randn_like(x)
        n = x.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x.device).unsqueeze(1)
        x = (x - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
             net(x, t_tensor))
        x = x / torch.sqrt(self.alphas[t])
        x = x + noise * torch.sqrt(self.betas[t])
        return x


# hyperparameters
batch_size = 512
n_epochs = 30


def train(ddpm: DDPM, net, device='cuda', ckpt_path='dldemos/ddpm/model.pth'):
    print('batch size:', batch_size)
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, -1))
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


def sample_img(ddpm: DDPM, net, n_sample=4, device='cuda'):
    shape = (n_sample, *get_img_shape())  # 1, 3, 28, 28
    res = ddpm.sample_backward(shape, net, device)
    return res


# CUDA_VISIBLE_DEVICES=1 nohup python -u dldemos/ddpm/main.py >> nohup_convbig_1000.txt 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python -u dldemos/ddpm/main.py

configs = [convnet_small_cfg, convnet_medium_cfg, convnet_big_cfg, unet_1_cfg]

if __name__ == '__main__':
    n_steps = 1000
    config_id = 2
    device = 'cuda'
    model_path = 'dldemos/ddpm/model_convbig_1000.pth'

    config = configs[config_id]
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    # train(ddpm, net, device=device, ckpt_path=model_path)

    n_sample = 81

    net.load_state_dict(torch.load(model_path))
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        imgs = sample_img(ddpm, net, n_sample=n_sample,
                          device=device).detach().cpu()
        for i in range(len(imgs)):
            minv, maxv = torch.min(imgs[i]), torch.max(imgs[i])
            imgs[i] = (imgs[i] - minv) / (maxv - minv) * 255
        imgs = einops.rearrange(imgs,
                                "(b1 b2) c h w -> (b1 h) (b2 w) c",
                                b1=int(n_sample**0.5))

        imgs = imgs.numpy().astype(np.uint8)

        cv2.imwrite('work_dirs/diffusion.jpg', imgs)