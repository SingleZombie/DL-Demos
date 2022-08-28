import os
import re


def read_imdb(dir='data/aclImdb', split='pos', is_train=True):
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    lines = []
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'rb') as f:
            line = f.read().decode('utf-8')
            lines.append(line)
    return lines


def read_imdb_words(dir='data/aclImdb',
                    split='pos',
                    is_train=True,
                    n_files=1000):
    subdir = 'train' if is_train else 'test'
    dir = os.path.join(dir, subdir, split)
    all_str = ''
    for file in os.listdir(dir):
        if n_files <= 0:
            break
        with open(os.path.join(dir, file), 'rb') as f:
            line = f.read().decode('utf-8')
            all_str += line
        n_files -= 1

    words = re.sub(u'([^\u0020\u0061-\u007a])', '', all_str.lower()).split(' ')

    return words


def read_imdb_vocab(dir='data/aclImdb'):
    fn = os.path.join(dir, 'imdb.vocab')
    with open(fn, 'rb') as f:
        word = f.read().decode('utf-8').replace('\n', ' ')
        words = re.sub(u'([^\u0020\u0061-\u007a])', '',
                       word.lower()).split(' ')
        filtered_words = [w for w in words if len(w) > 0]

    return filtered_words


def main():
    vocab = read_imdb_vocab()
    print(vocab[0])
    print(vocab[1])

    lines = read_imdb()
    print('Length of the file:', len(lines))
    print('lines[0]:', lines[0])
    words = read_imdb_words(n_files=100)
    print('Length of the words:', len(words))
    for i in range(5):
        print(words[i])


if __name__ == '__main__':
    main()
