# import os
# import time

# import torch
# import torch.distributed as dist
# import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel

# from dldemos.Transformer.model import Transformer
# from dldemos.Transformer.preprocess_data import (PAD_ID, get_dataloader,
#                                                  load_sentences, load_vocab)

# # Config
# batch_size = 64
# lr = 0.0001
# d_model = 512
# d_ff = 2048
# n_layers = 6
# heads = 8

# n_epochs = 40

# def reduce_mean(tensor, nprocs):
#     rt = tensor.clone()
#     dist.all_reduce(rt, op=dist.ReduceOp.SUM)
#     rt /= nprocs
#     return rt

# def main():
#     dist.init_process_group('nccl')
#     rank = dist.get_rank()
#     device_id = rank % torch.cuda.device_count()

#     en_vocab, zh_vocab = load_vocab()

#     en_train, zh_train, en_valid, zh_valid = load_sentences()
#     dataloader_train, sampler = get_dataloader(en_train, zh_train,
# batch_size,
#                                                True)
#     dataloader_valid = get_dataloader(en_valid, zh_valid)

#     print_interval = 1000

#     model = Transformer(len(en_vocab), len(zh_vocab), PAD_ID, d_model, d_ff,
#                         n_layers, heads)
#     model.to(device_id)

#     model = DistributedDataParallel(model, device_ids=[device_id])
#     optimizer = torch.optim.Adam(model.parameters(), lr)

#     # Optional: load model
#     ckpt_path = 'dldemos/Transformer/model_latest.pth'
#     optim_path = 'dldemos/Transformer/optimizer_latest.pth'
#     if os.path.exists(ckpt_path) and os.path.exists(optim_path):
#         map_location = {'cuda:0': f'cuda:{device_id}'}
#         state_dict = torch.load(ckpt_path, map_location=map_location)
#         model.module.load_state_dict(state_dict)
#         state_dict = torch.load(optim_path, map_location=map_location)
#         optimizer.load_state_dict(state_dict)
#         begin_epoch = int(
#             os.path.split(
#                 os.readlink(ckpt_path))[-1].split('.')[0].split('_')[1]) + 1
#     else:
#         begin_epoch = 0

#     citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
#     tic = time.time()
#     cnter = 0
#     dataset_len = len(dataloader_train.dataset)
#     if device_id == 0:
#         print('Dataset size:', dataset_len)
#     for epoch in range(begin_epoch, n_epochs):
#         sampler.set_epoch(epoch)

#         for x, y in dataloader_train:
#             x, y = x.to(device_id), y.to(device_id)
#             x_mask = x == PAD_ID
#             y_mask = y == PAD_ID
#             y_input = y[:, :-1]
#             y_label = y[:, 1:]
#             y_mask = y_mask[:, :-1]
#             y_hat = model(x, y_input, x_mask, y_mask)
#             n, seq_len = y_label.shape
#             y_hat = torch.reshape(y_hat, (n * seq_len, -1))
#             y_label = torch.reshape(y_label, (n * seq_len, ))
#             loss = citerion(y_hat, y_label)

#             y_label_mask = y_label != PAD_ID
#             preds = torch.argmax(y_hat, -1)
#             correct = preds == y_label
#             acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
#             optimizer.step()
#             loss = reduce_mean(loss, dist.get_world_size())
#             if device_id == 0:
#                 toc = time.time()
#                 interval = toc - tic
#                 minutes = int(interval // 60)
#                 seconds = int(interval % 60)
#                 if cnter % print_interval == 0:
#                     print(f'{cnter:08d} {minutes:02d}:{seconds:02d}'
#                           f' loss: {loss.item()} acc: {acc.item()}')
#             cnter += 1

#         if device_id == 0:
#             latest_model = 'dldemos/Transformer/model_latest.pth'
#             latest_optimizer = 'dldemos/Transformer/optimizer_latest.pth'
#             model_file = f'dldemos/Transformer/model_{epoch}.pth'
#             optim_file = f'dldemos/Transformer/optimizer_{epoch}.pth'
#             torch.save(model.module.state_dict(), model_file)
#             torch.save(optimizer.state_dict(), optim_file)

#             if os.path.exists(latest_model):
#                 os.remove(latest_model)
#             if os.path.exists(latest_optimizer):
#                 os.remove(latest_optimizer)

#             os.symlink(os.path.abspath(model_file), latest_model)
#             os.symlink(os.path.abspath(optim_file), latest_optimizer)

#             print(f'Model saved to {model_file}')

#         dist.barrier()

#         # if valid_period

#     print('Done.')

#     dist.destroy_process_group()

# if __name__ == '__main__':
#     main()

# # nohup bash dldemos/Transformer/dist_train.sh &
