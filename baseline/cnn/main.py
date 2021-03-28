import paddlehub as hub
import pandas as pd

model_list = ['senta_lstm']


def test(model_name):
    test_df = pd.read_csv("data/train_data.csv", encoding='utf-8')
    test_text = test_df["text"].to_list()
    test_label = test_df["label"]
    print("Loaded data.")

    senta = hub.Module(name=model_name)
    print("Loaded %s model: " % model_name)

    results = senta.sentiment_classify(texts=test_text)
    acc = 0
    for i in range(len(results)):
        if results[i]['sentiment_key'] == test_label[i]:
            acc += 1
    print("模型：%s\n测试集正确率：%.3f" % (model_name, acc/len(results)))


def prediction(model_name):
    pre_df = pd.read_csv("data/test1_data.csv", encoding='utf-8')
    pre_text = pre_df["text"].to_list()
    print("Loaded data.")

    senta = hub.Module(name=model_name)
    print("Loaded %s model: " % model_name)

    results = senta.sentiment_classify(texts=pre_text)
    answer_path = "answer_" + model_name + ".txt"
    with open(answer_path, "w", encoding='utf-8') as f:
        for result in results:
            f.write(str(result['sentiment_label']) + '\n')
    print("Answer saved to ==> %s" % answer_path)


def main(model_name):
    if model_name not in model_list:
        print("Unknown model name. Please check the model list: " + str(model_list))
        return

    test(model_name)
    prediction(model_name)


if __name__ == '__main__':
    main("senta_gru")
