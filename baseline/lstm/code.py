import jieba
sentence = "I am a dog"
sentence = sentence.split(" ")
print(type(sentence))

a = "我是中国人"
a = list(jieba.cut(a))
print(a)
print(type(a))