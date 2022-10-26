import json
import numpy as np
from collections import Counter

from torchtext.data import get_tokenizer


def read_file(json_path):
    english_sentences = []
    chinese_sentences = []
    tokenizer = get_tokenizer('basic_english')
    with open(json_path, 'r') as fp:
        for line in fp:
            line = json.loads(line)
            english, chinese = line['english'], line['chinese']
            # Correct mislabeled data
            if not english.isascii():
                english, chinese = chinese, english
            # Tokenize (Chinese characters are treated as words)
            english = tokenizer(english)
            english_sentences.append(english)
            chinese_sentences.append(chinese)
    return english_sentences, chinese_sentences


def create_vocab(sentences, max_element=None):
    char_set = Counter()
    for sentence in sentences:
        c_set = Counter(sentence)
        char_set.update(c_set)

    if max_element is None:
        return list(char_set.keys())
    else:
        words_freq = char_set.most_common(max_element)
        # pair array to double array
        words, freq = zip(*words_freq)
        return list(words)


def main():
    en_sens, zh_sens = read_file(
        'data/translation2019zh/translation2019zh_valid.json')
    en_vocab = create_vocab(en_sens)
    zh_vocab = create_vocab(zh_sens)
    print(len(en_vocab))
    print(len(zh_vocab))


if __name__ == '__main__':
    main()