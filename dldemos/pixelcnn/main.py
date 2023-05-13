from dldemos.pixelcnn.dataset import get_dataloader, get_img_shape
from dldemos.pixelcnn.model import PixelCNN

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import einops
import cv2

import numpy as np

batch_size = 128
n_class = 4


def train(model, device, model_path):
    dataloader = get_dataloader(batch_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    n_epochs = 10
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            y = torch.ceil(x * (n_class - 1)).long()
            y = y.squeeze(1)
            predict_y = model(x)
            loss = loss_fn(predict_y, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), model_path)
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')


def sample(model, device, model_path, output_path, n_sample=81):

    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    C, H, W = get_img_shape()  # (1, 28, 28)
    x = torch.zeros((n_sample, C, H, W)).to(device)
    with torch.no_grad():
        for i in range(H):
            for j in range(W):
                output = model(x)
                prob_dist = F.softmax(output[:, :, i, j], -1)
                pixel = torch.multinomial(prob_dist, 1).float() / (n_class - 1)
                x[:, :, i, j] = pixel

    imgs = x * 255
    imgs = imgs.clamp(0, 255)
    imgs = einops.rearrange(imgs,
                            '(b1 b2) c h w -> (b1 h) (b2 w) c',
                            b1=int(n_sample**0.5))

    imgs = imgs.detach().cpu().numpy().astype(np.uint8)

    cv2.imwrite(output_path, imgs)


if __name__ == '__main__':
    model = PixelCNN(64, 32, n_class)
    device = 'cuda'
    model_path = 'dldemos/pixelcnn/model.pth'
    #train(model, device, model_path)
    sample(model, device, model_path, 'work_dirs/pixelcnn.jpg')