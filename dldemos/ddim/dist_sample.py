import argparse
import os

import cv2
import einops
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from dldemos.ddim.configs import configs
from dldemos.ddim.ddim import DDIM
from dldemos.ddim.ddpm import DDPM
from dldemos.ddim.network import UNet


def sample_imgs(ddpm,
                net,
                output_dir,
                img_shape,
                n_sample=30000,
                device=0,
                simple_var=True,
                to_bgr=False,
                **kwargs):
    if img_shape[1] >= 256:
        max_batch_size = 16
    elif img_shape[1] >= 128:
        max_batch_size = 64
    else:
        max_batch_size = 256
    n_devices = dist.get_world_size()

    net = net.to(device)
    net = net.eval()

    os.makedirs(output_dir, exist_ok=True)

    index = 0
    with torch.no_grad():
        while index < n_sample:
            start_index = index + device * max_batch_size
            end_index = min(n_sample, index + (device + 1) * max_batch_size)

            local_batch_size = end_index - start_index
            if local_batch_size > 0:
                shape = (local_batch_size, *img_shape)
                imgs = ddpm.sample_backward(shape,
                                            net,
                                            device=device,
                                            simple_var=simple_var,
                                            **kwargs).detach().cpu()
                imgs = (imgs + 1) / 2 * 255
                imgs = imgs.clamp(0, 255).to(torch.uint8)

                img_list = einops.rearrange(imgs, 'n c h w -> n h w c').numpy()
                for i, img in enumerate(img_list):
                    if to_bgr:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f'{output_dir}/{i+start_index}.jpg', img)

            index += max_batch_size * n_devices


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
    map_location = {'cuda:0': f'cuda:{device}'}
    resume_path = model_path
    state_dict = torch.load(resume_path, map_location=map_location)
    net.module.load_state_dict(state_dict)

    ddim = DDIM(device, n_steps)
    sample_imgs(ddpm,
                net,
                'work_dirs/diffusion_ddpm_sigma_hat',
                img_shape,
                device=device,
                to_bgr=to_bgr)
    dist.barrier()
    sample_imgs(ddim,
                net,
                'work_dirs/diffusion_ddpm_eta_0',
                img_shape,
                device=device,
                to_bgr=to_bgr,
                ddim_step=1000,
                simple_var=False,
                eta=0)
    dist.barrier()

    dist.destroy_process_group()

# torchrun --nproc_per_node=8 dldemos/ddim/dist_sample.py -c 2 \
#   > work_dirs/tmp.txt
