import argparse
import ast

import paddle
import paddlehub as hub
import pandas as pd
from dataset import MyDataset

parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=False,
                    help="Whether use GPU for finetuning, input should be True or False")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
args = parser.parse_args()


def main():
    # Step1: load Paddlehub senta pretrained model
    model = hub.Module(name='ernie_tiny', version='2.0.1', task='seq-cls', num_classes=2)
    tokenizer = model.get_tokenizer()

    # Step2: 加载数据集
    train_dataset = MyDataset(tokenizer, mode="train")
    dev_dataset = MyDataset(tokenizer, mode="dev")
    test_dataset = MyDataset(tokenizer, mode="test")

    # Setup feed list for data feeder
    optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls', use_gpu=True)

    trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset)

    # 在测试集上评估当前训练模型
    trainer.evaluate(test_dataset, batch_size=32)


def test():
    test_df = pd.read_csv("data/train_data.csv", encoding='utf-8')
    test_text = test_df["text"].to_list()
    test_label = test_df["label"].to_list()
    label_map = {0: 'negative', 1: 'positive'}

    model = hub.Module(
        name='ernie_tiny',
        version='2.0.1',
        task='seq-cls',
        load_checkpoint='./test_ernie_text_cls/best_model/model.pdparams',
        label_map=label_map)

    results = model.predict(test_text, max_seq_len=50, batch_size=1, use_gpu=False)
    acc = 0
    for idx, text in enumerate(test_text):
        print('Data: {} \t Label: {}'.format(text[0], results[idx]))
        if results[idx] == test_label[idx]:
            acc += 1

    print("测试集正确率：%.3f" % acc / len(test_text))


def eval():
    eval_df = pd.read_csv("data/test1_data.csv", encoding='utf-8')
    eval_text = eval_df.to_list()
    label_map = {0: 'negative', 1: 'positive'}

    model = hub.Module(
        name='ernie_tiny',
        version='2.0.1',
        task='seq-cls',
        load_checkpoint='./test_ernie_text_cls/best_model/model.pdparams',
        label_map=label_map)

    results = model.predict(eval_text, max_seq_len=50, batch_size=1, use_gpu=False)
    with open("answer.txt", 'w') as f:
        for idx, text in enumerate(eval_text):
            print('Data: {} \t Label: {}'.format(text[0], results[idx]))
            if results[idx] == "negative":
                f.write(str(0) + '\n')
            else:
                f.write(str(1) + '\n')


if __name__ == '__main__':
    main()
    test()
    eval()
