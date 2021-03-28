import paddlehub as hub
import pandas as pd
TYPE="test"
with open("data/test.txt","w") as f:
    f.write("label\ttext_a\n")
f.close()

data = pd.read_csv("data/test.csv", encoding='utf-8')
text = data["text"].to_list()
label = data["label"].to_list()
size = data.shape[0]

for i in range(0,size):
    if(label[i] == 1):
        label_text = "positive"
    else:
        label_text = "negative"
    word = "{}\t{}\n".format(label_text, text[i])
    with open("data/test.txt","a+") as f:
        f.write(word)
    print(word)

f.close()
