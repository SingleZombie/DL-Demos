import os

from torchtext.data import get_tokenizer


def read_imdb(dir='data/aclImdb', split='pos', is_train=True):
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    lines = []
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'rb') as f:
            line = f.read().decode('utf-8')
            lines.append(line)
    return lines


def main():
    lines = read_imdb()
    print('Length of the file:', len(lines))
    print('lines[0]:', lines[0])
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(lines[0])
    print('lines[0] tokens:', tokens)


if __name__ == '__main__':
    main()
