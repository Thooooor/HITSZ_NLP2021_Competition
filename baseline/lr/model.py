import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline


def load_stopwords_list(list_name='hit'):
    list_path = "stopwords/%s_stopwords.txt" % list_name

    with open(list_path, encoding='utf-8') as f:
        lines = f.readlines()
        result = [i.strip('\n') for i in lines]

    return result


def main():
    train_df = pd.read_csv("data/all_data.csv", encoding='utf-8')
    test_df = pd.read_csv("data/train_data.csv", encoding='utf-8')
    train_text = train_df["text_a"].to_list()
    train_label = train_df["label"].to_list()
    test_text = test_df["text"].to_list()
    test_label = test_df["label"].to_list()

    X_train = train_text
    X_test = test_text
    y_train = train_label
    y_test = test_label

    stop_words = load_stopwords_list('hit')

    vect = CountVectorizer(max_df=0.8, min_df=3, stop_words=stop_words, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')
    vect.fit(train_text)

    pipe = make_pipeline(TfidfVectorizer(min_df=3, stop_words=stop_words, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b'),
                         LogisticRegression())
    pipe.fit(X_train, y_train)
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print('平均交叉验证准确率：{:.3f}'.format(np.mean(scores)))
    print('测试集准确率：{:.3f}'.format(pipe.score(X_test, y_test)))

    final_test = pd.read_csv("data/test1_data.csv")['text'].to_list()
    y_pre = pipe.predict(final_test)
    with open("answer.txt", 'w') as f:
        for y in y_pre:
            f.write(str(y))
            f.write('\n')


if __name__ == '__main__':
    main()
