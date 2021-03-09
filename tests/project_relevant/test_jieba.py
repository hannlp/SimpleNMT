import jieba

sentence = "我爱你，我亲爱的祖国母亲。"

word_list = list(jieba.cut(sentence))

print(word_list)