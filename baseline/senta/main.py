from senta import Senta
import pandas as pd

my_senta = Senta()
use_cuda = False
my_senta.init_model(model_class="ernie_1.0_skep_large_ch", task="sentiment_classify", use_cuda=use_cuda)

test_df = pd.read_csv("data/train_data.csv", encoding='utf-8')
pre_df = pd.read_csv("data/test1_data.csv", encoding='utf-8')
test_text = test_df["text"]
test_label = test_df["label"]
pre_text = pre_df["text"]
print("Loaded data.")

pre_result = my_senta.predict(pre_text)

with open("answer.txt", 'w', encoding="utf-8") as f:
    for result in pre_result:
        print(result)
        if result[1] == "positive":
            f.write(str(1))
        else:
            f.write(str(0))
        f.write('\n')

print("Done.")
