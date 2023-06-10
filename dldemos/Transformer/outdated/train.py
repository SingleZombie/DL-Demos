# import time

# import numpy as np
# import torch
# import torch.nn as nn

# from dldemos.Transformer.dataset import tensor_to_sentence
# from dldemos.Transformer.model import Transformer
# from dldemos.Transformer.preprocess_data import (PAD_ID, get_dataloader,
#                                                  load_sentences, load_vocab)

# # Config
# batch_size = 64
# lr = 0.0001
# d_model = 512
# d_ff = 1024
# n_layers = 6
# heads = 8

# def main():
#     en_vocab, zh_vocab = load_vocab()

#     en_train, zh_train, en_valid, zh_valid = load_sentences()
#     dataloader_train = get_dataloader(en_train, zh_train, batch_size)

#     print_interval = 1000
#     device_id = 0

#     model = Transformer(len(en_vocab), len(zh_vocab), d_model, d_ff,
# n_layers,
#                         heads)
#     model.to(device_id)

#     model.init_weights()

#     optimizer = torch.optim.Adam(model.parameters(), lr)
#     citerion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
#     tic = time.time()
#     cnter = 0
#     dataset_len = len(dataloader_train.dataset)
#     print('Dataset size:', dataset_len)
#     for epoch in range(10):
#         loss_sum = 0

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

#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
#             optimizer.step()

#             loss_sum += loss.item()

#             toc = time.time()
#             interval = toc - tic
#             minutes = int(interval // 60)
#             seconds = int(interval % 60)
#             if cnter % print_interval == 0:
#                 print(f'{cnter:08d} {minutes:02d}:{seconds:02d}'
#                       f' loss: {loss.item()}')
#             cnter += 1

#         print(f'Epoch {epoch}. loss: {loss_sum / dataset_len}')

#     torch.save(model.state_dict(), 'dldemos/Transformer/model.pth')
#     print('Done.')
