import heapq
import numpy as np
from collections import defaultdict
import re

startToken = '<S>'
endToken = '</S>'
unkToken = '<UNK>'
padToken = '<PAD>'


def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_vocab(corpus):
    vocab = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            vocab[f"{' '.join(list(word))}</w>"] += 1
    return vocab


def byte_pair_encoding(data, n = 10):
    vocab = get_vocab(data)
    path = []
    for i in range(n):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        path.append(best)
        vocab = merge_vocab(best, vocab)
    return path


def load_subword_nmt_table(path):
    table = dict()
    cur_priority = 1
    for line in path:
        if '#version' in line:
            continue
        token_1, token_2 = line
        table[(token_1, token_2)] = int(cur_priority)
        cur_priority += 1
    return table


def load_merge_table(path):
    table = dict()
    with open(path) as f:
        for line in f:
            token_1, token_2, priority = line.split('\t')
            table[(token_1, token_2)] = int(priority)

    return table


def tokenize_word(merge_rules, word, dropout=0.0,
                  random_generator=np.random.RandomState(),
                  sentinels=['^', '$'],
                  regime='begin',
                  bpe_symbol='`',
                  always_merge_sentinels=True):
    sw_tokens = list(word)

    if always_merge_sentinels:
        sw_tokens = [sentinels[0] + sw_tokens[0]] + sw_tokens[1:]
        sw_tokens = sw_tokens[:-1] + [sw_tokens[-1] + sentinels[1]]
    else:
        beg_sentinel = [sentinels[0]] if len(sentinels[0]) > 0 else []
        end_sentinel = [sentinels[1]] if len(sentinels[1]) > 0 else []
        sw_tokens = beg_sentinel + sw_tokens + end_sentinel

    merge_heap = []

    for pos in range(len(sw_tokens) - 1):
        cur_nxt_pair = (sw_tokens[pos], sw_tokens[pos + 1])
        if cur_nxt_pair in merge_rules:
            cur_priority = merge_rules[cur_nxt_pair]
            merge_heap.append([cur_priority, pos])

    heapq.heapify(merge_heap)

    sw_length = len(sw_tokens)
    dropped_merges = []

    while merge_heap:
        cur_priority, cur_pos = heapq.heappop(merge_heap)

        if cur_pos > sw_length - 2:
            continue
        cur = sw_tokens[cur_pos]
        nxt = sw_tokens[cur_pos + 1]

        if merge_rules.get((cur, nxt), None) != cur_priority:
            continue

        if random_generator.rand() < dropout:
            dropped_merges.append([cur_priority, cur_pos])
            continue

        sw_tokens[cur_pos:cur_pos + 2] = [cur + nxt]
        sw_length -= 1

        for pair in merge_heap:
            if pair[1] > cur_pos:
                pair[1] -= 1

        for priority, position in dropped_merges:
            if position > cur_pos:
                position -= 1
            heapq.heappush(merge_heap, [priority, position])

        dropped_merges = []

        # Add new possible merge
        new_cur = sw_tokens[cur_pos]
        if cur_pos > 0:
            prev = sw_tokens[cur_pos - 1]
            if (prev, new_cur) in merge_rules:
                heapq.heappush(merge_heap, [merge_rules[(prev, new_cur)], cur_pos - 1])

        if cur_pos < (sw_length - 1):
            new_next = sw_tokens[cur_pos + 1]
            if (new_cur, new_next) in merge_rules:
                heapq.heappush(merge_heap, [merge_rules[(new_cur, new_next)], cur_pos])

    sw_tokens[0] = sw_tokens[0].replace(sentinels[0], '')
    sw_tokens[-1] = sw_tokens[-1].replace(sentinels[1], '')

    if regime == 'begin':
        for i in range(1, sw_length):
            sw_tokens[i] = bpe_symbol + sw_tokens[i]

        if sw_tokens[0] == '':
            sw_tokens = sw_tokens[1:]
            sw_tokens[0] = sw_tokens[0].lstrip(bpe_symbol)
        if sw_tokens[-1] == bpe_symbol:
            sw_tokens.pop()
    elif regime == 'end':
        for i in range(sw_length - 1):
            sw_tokens[i] = sw_tokens[i] + bpe_symbol
        if sw_tokens[0] == bpe_symbol:
            sw_tokens.pop(0)
        if sw_tokens[-1] == '':
            sw_tokens = sw_tokens[:-1]
            sw_tokens[-1] = sw_tokens[-1].rstrip(bpe_symbol)

    return sw_tokens


def tokenize_text(rules, line, dropout=0.0, random_generator=np.random.RandomState(), **args):
    return ' '.join(
        [' '.join(tokenize_word(rules, word, dropout, random_generator, **args)) for word in line.split(' ')])


class BpeOnlineTokenizer:
    def __init__(self, bpe_dropout_rate, merge_table, random_seed=None):
        self.random_generator = np.random.RandomState(random_seed)
        self.bpe_dropout_rate = bpe_dropout_rate
        self.merge_table = merge_table

    def __call__(self, line, **args):
        return tokenize_text(self.merge_table, line, self.bpe_dropout_rate, self.random_generator, **args)


class BpeOnlineParallelApplier:
    def __init__(self, bpe_dropout_rates, merge_tables, random_seed=42):
        assert len(bpe_dropout_rates) == len(merge_tables)
        self.bpe_appliers = []
        for rate, table in zip(bpe_dropout_rates, merge_tables):
            if table is not None:
                self.bpe_appliers.append(BpeOnlineTokenizer(rate, table, random_seed))
            else:
                self.bpe_appliers.append(lambda x: x)

    def __call__(self, lines):
        assert len(self.bpe_appliers) == len(lines)
        return tuple(applier(l) for applier, l in zip(self.bpe_appliers, lines))


def bpe(corpus, n, lang, dropout = 0.0):
    path = byte_pair_encoding(corpus, n)

    corpus_encoded = encode_corpus(corpus, path, lang, dropout)
    dictionary = defaultdict(int)

    for sentence in corpus_encoded:
        for pair in path:
            if pair[1][-4:] == '</w>':
                dictionary[f'{pair[0]}{pair[1][:-4]}'] += 1
            else:
                dictionary[f'{pair[0]}{pair[1]}##'] += 1
        for pair in sentence:
            dictionary[pair] += 1
            for char in pair:
                dictionary[char] += 1

    L = sorted([(pair, dictionary[pair]) for pair in dictionary], key=lambda x: x[1], reverse=True)
    pairs = [pair for pair, _ in L] + [unkToken] + [padToken]
    pair2ind = {p : i for i, p in enumerate(pairs)}

    return path, corpus_encoded, pair2ind


def encode_corpus(corpus, path, lang, dropout = 0.0):
    merge_table = load_subword_nmt_table(path)
    corpus_encoded = []

    subword_nmt_tokenizer = BpeOnlineTokenizer(
        bpe_dropout_rate=dropout,
        merge_table=merge_table
    )

    for sentence in corpus:
        if lang == 'eng':
            corpus_encoded.append(subword_nmt_tokenizer(' '.join(sentence), sentinels=['', '</w>'], regime='end',
                                                        bpe_symbol='##').split())
        else:
            corpus_encoded.append(
                [startToken] + subword_nmt_tokenizer(' '.join(sentence), sentinels=['', '</w>'], regime='end',
                                                     bpe_symbol='##').split() + [endToken])

    return corpus_encoded


def decode_corpus(corpus, lang):
    corpus_decoded = []

    for sentence in corpus:
        sentence_decoded = []
        word_decoded = ''
        if lang == 'eng':
            for token in sentence:
                if token[-2:] == '##':
                    word_decoded += token[:-2]
                else:
                    word_decoded += token
                    sentence_decoded.append(word_decoded)
                    word_decoded = ''
        else:
            for token in sentence[1:-1]:
                if token[-2:] == '##':
                    word_decoded += token[:-2]
                else:
                    word_decoded += token
                    sentence_decoded.append(word_decoded)
                    word_decoded = ''
        corpus_decoded.append(sentence_decoded)
    return corpus_decoded

