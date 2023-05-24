import torch
from torchtext.vocab import GloVe

glove = GloVe(name='6B', dim=100)

# Get vectors
tensor = glove.get_vecs_by_tokens(['', '1998', '199999998', ',', 'cat'], True)
print(tensor)

# Iterate the vocab
myvocab = glove.itos
print(len(myvocab))
print(myvocab[0], myvocab[1], myvocab[2], myvocab[3])


def get_counterpart(x1, y1, x2):
    """Find y2 that makes x1-y1=x2-y2."""
    x1_id = glove.stoi[x1]
    y1_id = glove.stoi[y1]
    x2_id = glove.stoi[x2]
    x1, y1, x2 = glove.get_vecs_by_tokens([x1, y1, x2], True)
    target = x2 - x1 + y1
    max_sim = 0
    max_id = -1
    for i in range(len(myvocab)):
        vector = glove.get_vecs_by_tokens([myvocab[i]], True)[0]
        cossim = torch.dot(target, vector)
        if cossim > max_sim and i not in {x1_id, y1_id, x2_id}:
            max_sim = cossim
            max_id = i
    return myvocab[max_id]


print(get_counterpart('man', 'woman', 'king'))
print(get_counterpart('more', 'less', 'long'))
print(get_counterpart('apple', 'red', 'banana'))
