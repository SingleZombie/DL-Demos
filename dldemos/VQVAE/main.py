import os
import time

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dldemos.VQVAE.model import VQVAE
from dldemos.pixelcnn.model import GatedPixelCNN
from dldemos.VQVAE.dataset import get_dataloader

# For MNIST
# dim = 256
# n_embedding = 512
# batch_size = 256
# n_epochs = 100
# l_w_embedding = 1
# l_w_commitment = 0.25 * l_w_embedding
# lr = 2e-4

# n_epochs_2 = 20
# pixelcnn_n_blocks = 15
# pixelcnn_dim = 128
# pixelcnn_linear_dim = 32

# For CelebA

#2


def train_vqvae(model: VQVAE,
                img_shape=None,
                device='cuda',
                ckpt_path='dldemos/VQVAE/model.pth',
                batch_size=64,
                dataset_type='MNIST',
                lr=1e-3,
                n_epochs=100,
                l_w_embedding=1,
                l_w_commitment=0.25):
    print('batch size:', batch_size)
    dataloader = get_dataloader(dataset_type, batch_size, img_shape=img_shape)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)

            x_hat, ze, zq = model(x)
            l_reconstruct = mse_loss(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)
            l_commitment = mse_loss(ze, zq.detach())
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def train_generative_model(vqvae: VQVAE,
                           model,
                           n_embedding,
                           device='cuda',
                           ckpt_path='dldemos/VQVAE/gen_model.pth',
                           dataset_type='MNIST',
                           batch_size=64,
                           n_epoch=50):
    dataloader = get_dataloader(dataset_type, batch_size)
    vqvae.to(device)
    vqvae.eval()
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    tic = time.time()
    for e in range(n_epoch):
        total_loss = 0
        for x in dataloader:
            current_batch_size = x.shape[0]
            with torch.no_grad():
                x = x.to(device)
                x = vqvae.encode(x)

            model_input = x.unsqueeze(1).float() / (n_embedding - 1)
            predict_x = model(model_input)
            loss = loss_fn(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), ckpt_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def reconstruct(model, x, device, dataset_type='MNIST'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'work_dirs/vqvae_reconstruct_{dataset_type}.jpg', x_cat)


def sample_imgs(vqvae: VQVAE,
                gen_model,
                img_shape,
                n_embedding,
                n_sample=81,
                device='cuda',
                dataset_type='MNIST'):
    vqvae = vqvae.to(device)
    vqvae.eval()
    gen_model = gen_model.to(device)
    gen_model.eval()

    C, H, W = img_shape
    H, W = vqvae.get_latent_HW((C, H, W))
    input_shape = (n_sample, 1, H, W)
    x = torch.zeros(input_shape).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = gen_model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist,
                                          1).float() / (n_embedding - 1)
                x[:, :, i, j] = pixel

    generated_latent = torch.ceil(x.squeeze(1) * (n_embedding - 1)).long()
    imgs = vqvae.decode(generated_latent)

    imgs = imgs * 255
    imgs = imgs.clip(0, 255)
    imgs = einops.rearrange(imgs,
                            '(n1 n2) c h w -> (n1 h) (n2 w) c',
                            n1=int(n_sample**0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA':
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'work_dirs/vqvae_sample_{dataset_type}.jpg', imgs)


mnist_cfg1 = dict()

celeba_cfg1 = dict(dataset_type='CelebA',
                   img_shape=(3, 64, 64),
                   dim=128,
                   n_embedding=64,
                   n_residual=3,
                   batch_size=128,
                   n_epochs=200,
                   l_w_embedding=1,
                   l_w_commitment=0.25,
                   lr=1e-3,
                   n_epochs_2=50,
                   batch_size_2=512,
                   pixelcnn_n_blocks=15,
                   pixelcnn_dim=128,
                   pixelcnn_linear_dim=32,
                   vqvae_path='dldemos/VQVAE/model_celeba.pth',
                   gen_model_path='dldemos/VQVAE/gen_model_celeba.pth')

celeba_cfg2 = dict(dataset_type='CelebA',
                   img_shape=(3, 128, 128),
                   dim=128,
                   n_embedding=64,
                   n_residual=3,
                   batch_size=64,
                   n_epochs=200,
                   l_w_embedding=1,
                   l_w_commitment=0.25,
                   lr=1e-3,
                   n_epochs_2=50,
                   batch_size_2=512,
                   pixelcnn_n_blocks=15,
                   pixelcnn_dim=128,
                   pixelcnn_linear_dim=32,
                   vqvae_path='dldemos/VQVAE/model_celeba_2.pth',
                   gen_model_path='dldemos/VQVAE/gen_model_celeba_2.pth')

celeba_cfg3 = dict(dataset_type='CelebA',
                   img_shape=(3, 128, 128),
                   dim=256,
                   n_embedding=128,
                   n_residual=2,
                   batch_size=32,
                   n_epochs=50,
                   l_w_embedding=1,
                   l_w_commitment=0.25,
                   lr=2e-4,
                   n_epochs_2=50,
                   batch_size_2=64,
                   pixelcnn_n_blocks=15,
                   pixelcnn_dim=128,
                   pixelcnn_linear_dim=32,
                   vqvae_path='dldemos/VQVAE/model_celeba_3.pth',
                   gen_model_path='dldemos/VQVAE/gen_model_celeba_3.pth')
cfgs = [celeba_cfg1, celeba_cfg2, celeba_cfg3]

if __name__ == '__main__':
    os.makedirs('work_dirs', exist_ok=True)
    cfg = cfgs[0]

    device = 'cuda'

    img_shape = cfg['img_shape']
    vqvae = VQVAE(img_shape[0], cfg['dim'], cfg['n_embedding'], 2,
                  cfg['n_residual'])
    gen_model = GatedPixelCNN(cfg['pixelcnn_n_blocks'], cfg['pixelcnn_dim'],
                              cfg['pixelcnn_linear_dim'], True,
                              cfg['n_embedding'])

    # 1. Train VQVAE
    # train_vqvae(vqvae,
    #             img_shape=(img_shape[1], img_shape[2]),
    #             device=device,
    #             ckpt_path=cfg['vqvae_path'],
    #             batch_size=cfg['batch_size'],
    #             dataset_type=cfg['dataset_type'],
    #             lr=cfg['lr'],
    #             n_epochs=cfg['n_epochs'],
    #             l_w_embedding=cfg['l_w_embedding'],
    #             l_w_commitment=cfg['l_w_commitment'])

    # 2. Test VQVAE by visualizaing reconstruction result
    vqvae.load_state_dict(torch.load(cfg['vqvae_path']))
    dataloader = get_dataloader(cfg['dataset_type'],
                                16,
                                img_shape=(img_shape[1], img_shape[2]))
    img = next(iter(dataloader)).to(device)
    reconstruct(vqvae, img, device, cfg['dataset_type'])

    # 3. Train Generative model (Gated PixelCNN in our project)
    # vqvae.load_state_dict(torch.load(vqvae_path))
    # train_generative_model(vqvae, gen_model, device, gen_model_path)

    # 4. Sample VQVAE
    # vqvae.load_state_dict(torch.load(vqvae_path))
    # gen_model.load_state_dict(torch.load(gen_model_path))
    # sample_imgs(vqvae, gen_model)
