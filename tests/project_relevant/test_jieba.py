import jieba

zh = "我爱你，我亲爱的祖国母亲。"

word_list1 = list(jieba.cut(zh))

en = 'in Europe , knowing that the current bank is explicitly to address \
    its capital shortfalls and leverage problems , and to treat residual assets with residual residual status.'

word_list2 = [w for w in list(jieba.cut(en)) if w.strip()]

print(word_list1)
print(word_list2)