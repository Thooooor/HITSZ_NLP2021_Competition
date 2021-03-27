import re
import random
import tarfile
import requests
import numpy as np
import paddle
import pandas as pd
import jieba
# from model import SentimentClassifier
# from paddle.nn import Embedding
TRAIN_DIR_PATH = "train.csv"
from paddle.nn import LSTM, Embedding, Dropout, Linear
import paddle.nn.functional as F

# 定义一个用于情感分类的网络实例，SentimentClassifier
class SentimentClassifier(paddle.nn.Layer):
    def __init__(self, hidden_size, vocab_size, class_num=2, num_steps=128, num_layers=1, init_scale=0.1, dropout=None):
        # 参数含义如下：
        # 1.hidden_size，表示embedding-size，hidden和cell向量的维度
        # 2.vocab_size，模型可以考虑的词表大小
        # 3.class_num，情感类型个数，可以是2分类，也可以是多分类
        # 4.num_steps，表示这个情感分析模型最大可以考虑的句子长度
        # 5.num_layers，表示网络的层数
        # 6.init_scale，表示网络内部的参数的初始化范围
        # 长短时记忆网络内部用了很多Tanh，Sigmoid等激活函数，这些函数对数值精度非常敏感，
        # 因此我们一般只使用比较小的初始化范围，以保证效果

        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout

        # 声明一个LSTM模型，用来把每个句子抽象成向量
        self.simple_lstm_rnn = LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers)

        # 声明一个embedding层，用来把句子中的每个词转换为向量
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, sparse=False,
                                   weight_attr=paddle.ParamAttr(
                                       initializer=paddle.nn.initializer.Uniform(low=-init_scale, high=init_scale)))

        # 在得到一个句子的向量表示后，需要根据这个向量表示对这个句子进行分类
        # 一般来说，可以把这个句子的向量表示乘以一个大小为[self.hidden_size, self.class_num]的W参数，
        # 并加上一个大小为[self.class_num]的b参数，从而达到把句子向量映射到分类结果的目的

        # 我们需要声明最终在使用句子向量映射到具体情感类别过程中所需要使用的参数
        # 这个参数的大小一般是[self.hidden_size, self.class_num]
        self.cls_fc = Linear(in_features=self.hidden_size, out_features=self.class_num,
                             weight_attr=None, bias_attr=None)
        self.dropout_layer = Dropout(p=self.dropout, mode='upscale_in_train')

    def forward(self, input, label):
        # 首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        init_hidden_data = np.zeros(
            (self.num_layers, batch_size, embedding_size), dtype='float32')
        init_cell_data = np.zeros(
            (self.num_layers, batch_size, embedding_size), dtype='float32')

        # 将这些初始记忆转换为飞桨可计算的向量
        # 设置stop_gradient=True，避免这些向量被更新，从而影响训练效果
        init_hidden = paddle.to_tensor(init_hidden_data)
        init_hidden.stop_gradient = True
        init_cell = paddle.to_tensor(init_cell_data)
        init_cell.stop_gradient = True

        init_h = paddle.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])
        init_c = paddle.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        # 将输入的句子的mini-batch转换为词向量表示
        x_emb = self.embedding(input)
        x_emb = paddle.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = self.dropout_layer(x_emb)

        # 使用LSTM网络，把每个句子转换为向量表示
        rnn_out, (last_hidden, last_cell) = self.simple_lstm_rnn(x_emb, (init_h, init_c))
        last_hidden = paddle.reshape(
            last_hidden[-1], shape=[-1, self.hidden_size])

        # 将每个句子的向量表示映射到具体的情感类别上
        projection = self.cls_fc(last_hidden)
        pred = F.softmax(projection, axis=-1)

        # 根据给定的标签信息，计算整个网络的损失函数，这里我们可以直接使用分类任务中常使用的交叉熵来训练网络
        loss = F.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = paddle.mean(loss)

        # 最终返回预测结果pred，和网络的loss
        return pred, loss

def load_imdb(is_training):
    data_set = []
    # 我们把数据依次读取出来，并放到data_set里
    # data_set中每个元素都是一个二元组，（句子，label），其中label=0表示负向情感，label=1表示正向情感

    train_df = pd.read_csv(TRAIN_DIR_PATH, encoding='utf-8')
    train_text = train_df["text"]
    train_label = train_df["label"]
    size = train_df.shape[0]
    for i in range(0, size):
        sentence = train_text[i]
        sentence_label = train_label[i]
        data_set.append((sentence, sentence_label))
    return data_set


def data_preprocess(corpus):
    """进行切词"""
    data_set = []
    for sentence, sentence_label in corpus:
        sentence = list(jieba.cut(sentence))
        data_set.append((sentence, sentence_label))
    return data_set

"""在经过切词之后，需要构造一个词典，把每个词转化为一个ID，用来训练神经网络"""
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
        sentence = [word2id_dict[word] if word in word2id_dict \
                    else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


# 编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
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

def train():
    step = 0
    sentiment_classifier = SentimentClassifier(
        embedding_size, vocab_size, num_steps=max_seq_len, num_layers=1)
    # 创建优化器Optimizer，用于更新这个网络的参数
    optimizer = paddle.optimizer.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999,
                                      parameters=sentiment_classifier.parameters())

    sentiment_classifier.train()
    for sentences, labels in build_batch(
            word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len):

        sentences_var = paddle.to_tensor(sentences)
        labels_var = paddle.to_tensor(labels)
        pred, loss = sentiment_classifier(sentences_var, labels_var)

        # 后向传播
        loss.backward()
        # 最小化loss
        optimizer.step()
        # 清除梯度
        optimizer.clear_grad()

        step += 1
        if step % 100 == 0:
            print("step %d, loss %.3f" % (step, loss.numpy()[0]))

    # 我们希望在网络训练结束以后评估一下训练好的网络的效果
    # 通过eval()函数，将网络设置为eval模式，在eval模式中，网络不会进行梯度更新
    eval(sentiment_classifier)


def eval(sentiment_classifier):
    sentiment_classifier.eval()
    # 这里我们需要记录模型预测结果的准确率
    # 对于二分类任务来说，准确率的计算公式为：
    # (true_positive + true_negative) /
    # (true_positive + true_negative + false_positive + false_negative)
    tp = 0.
    tn = 0.
    fp = 0.
    fn = 0.
    for sentences, labels in build_batch(
            word2id_dict, test_corpus, batch_size, 1, max_seq_len):

        sentences_var = paddle.to_tensor(sentences)
        labels_var = paddle.to_tensor(labels)

        # 获取模型对当前batch的输出结果
        pred, loss = sentiment_classifier(sentences_var, labels_var)

        # 把输出结果转换为numpy array的数据结构
        # 遍历这个数据结构，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
        pred = pred.numpy()
        for i in range(len(pred)):
            if labels[i][0] == 1:
                if pred[i][1] > pred[i][0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i][1] > pred[i][0]:
                    fp += 1
                else:
                    tn += 1

    # 输出最终评估的模型效果
    print("the acc in the test set is %.3f" % ((tp + tn) / (tp + tn + fp + fn)))

train_corpus = load_imdb(True)
test_corpus = load_imdb(False)
train_corpus = data_preprocess(train_corpus)
test_corpus = data_preprocess(test_corpus)
word2id_freq, word2id_dict = build_dict(train_corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(10), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))
train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
print("%d tokens in the corpus" % len(train_corpus))
print(train_corpus[:5])
print(test_corpus[:5])
for batch_id, batch in enumerate(build_batch(word2id_dict, train_corpus, batch_size=3, epoch_num=3, max_seq_len=30)):
    print(batch)

# 开始训练
batch_size = 16
epoch_num = 5
embedding_size = 32
learning_rate = 0.01
max_seq_len = 16
# 使用GPU
paddle.set_device('gpu:0')




train()