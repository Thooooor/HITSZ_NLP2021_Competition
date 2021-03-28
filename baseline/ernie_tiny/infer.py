import paddlehub as hub
import pandas as pd

def get_test_data():
    data = pd.read_csv("data/test1_data.csv", encoding='utf-8')
    text = data["text"].to_list()
    data_list = []
    for e in text:
        ele = []
        ele.append(e)
        data_list.append(ele)
    return data_list
data = get_test_data()
label_map = {0: 'negative', 1: 'positive'}
model = hub.Module(
    name='ernie_tiny',
    version='2.0.1',
    task='seq-cls',
    load_checkpoint='./test_ernie_text_cls/best_model/model.pdparams',
    label_map=label_map)
results = model.predict(data, max_seq_len=50, batch_size=1, use_gpu=False)

with open("anwser.txt","w",encoding='utf-8') as f:
    for idx, text in enumerate(data):
        if(results[idx] == "negative"):
            f.write("0\n")
        else:
            f.write("1\n")
        print('Data: {} \t Lable: {}'.format(text[0], results[idx]))
f.close()