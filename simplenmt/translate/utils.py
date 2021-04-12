from data.constants import Constants

def de_numericalize(vocab, tokens):
    #remove_constants={}
    remove_constants={
        Constants.PAD, Constants.START, Constants.END}
          
    sentences = []
    for sentence in tokens:
        end = False
        words_list = []
        for word_id in sentence:
            word = vocab.itos[word_id]
            end = True if word == Constants.END else end
            if word not in remove_constants and not end:
                words_list.append(word)
            else:
                pass
        sentences.append(words_list)

    return sentences