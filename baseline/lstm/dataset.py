import jieba
import jieba.analyse
import pandas as pd
import random
import numpy as np


def data_preprocess(corpus):
    data_set = []
    jieba.analyse.set_stop_words("stopwords/baidu_stopwords.txt")
    for sentence, sentence_label in corpus:
        # 这里有一个小trick是把所有的句子转换为小写，从而减小词表的大小
        # 一般来说这样的做法有助于效果提升
        sentence = jieba.analyse.extract_tags(sentence, topK=20)
        data_set.append((sentence, sentence_label))

    return data_set


# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        # 这里需要注意，一般来说我们可能需要查看一下test-set中，句子oov的比例，
        # 如果存在过多oov的情况，那就说明我们的训练数据不足或者切分存在巨大偏差，需要调整
        sentence = [word2id_dict[word] if word in word2id_dict else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


def build_batch(word2id_dict, corpus, batch_size, epoch_num, max_seq_len, shuffle=True, drop_last=True):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):

        # 每个epoch前都shuffle一下数据，有助于提高模型训练的效果
        # 但是对于预测任务，不要做数据shuffle
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
        if not drop_last and len(sentence_batch) > 0:
            yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


train_df = pd.read_csv("data/all_data.csv", encoding='utf-8')
test_df = pd.read_csv("data/train_data.csv", encoding='utf-8')
train_text = train_df["text_a"].to_list()
train_label = train_df["label"].to_list()
test_text = test_df["text"].to_list()
test_label = test_df["label"].to_list()

train_corpus = []
test_corpus = []

for i in range(len(train_text)):
    if train_label[i] == "positive":
        label = 1
    else:
        label = 0
    train_corpus.append((train_text[i], label))
print(train_corpus[:5])

for i in range(len(test_text)):
    if test_label[i] == "positive":
        label = 1
    else:
        label = 0
    test_corpus.append((test_text[i], label))
print(test_corpus[:5])

# 构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
word2id_freq, word2id_dict = build_dict(train_corpus)
vocab_size = len(word2id_freq)
print("there are totally %d different words in the corpus" % vocab_size)
# 把语料转换为id序列
train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
print("%d tokens in the corpus" % len(train_corpus))
print(train_corpus[:5])
print(test_corpus[:5])
