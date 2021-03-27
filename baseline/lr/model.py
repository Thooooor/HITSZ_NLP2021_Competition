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
    train_df = pd.read_csv("data/train.csv", encoding='utf-8')
    test_df = pd.read_csv("data/test.csv", encoding='utf-8')
    train_text = pd.read_csv("data/all.csv", encoding='utf-8')["text"]
    unlabel_text = pd.read_csv("data/unlabel_data.csv", encoding="utf-8")["text"]
    all_text = pd.concat([train_text, unlabel_text], axis=0, ignore_index=True)

    X_train = train_df["text"]
    X_test = test_df['text']
    y_train = train_df["label"]
    y_test = test_df['label']

    stop_words = load_stopwords_list('hit')

    vect = CountVectorizer(max_df=0.8, min_df=3, stop_words=stop_words, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b')
    vect.fit(all_text)

    lr = LogisticRegression()
    lr.fit(vect.transform(X_train), y_train)
    print('测试集准确率：{:.3f}'.format(lr.score(vect.transform(X_test), y_test)))

    pipe = make_pipeline(TfidfVectorizer(min_df=3, stop_words=stop_words, token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b'),
                         LogisticRegression())
    pipe.fit(X_train, y_train)
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    print('平均交叉验证准确率：{:.3f}'.format(np.mean(scores)))
    print('测试集准确率：{:.3f}'.format(pipe.score(X_test, y_test)))

    final_test = pd.read_csv("data/test1_data.csv")['text']
    y_pre = lr.predict(vect.transform(final_test))
    with open("answer.txt", 'w') as f:
        for y in y_pre:
            f.write(str(y))
            f.write('\n')


if __name__ == '__main__':
    main()
