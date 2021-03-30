from dataset import *
from model import *
import random


# 使用GPU
# paddle.set_device('gpu:0')


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


train()
