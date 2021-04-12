from data.constants import Constants

def de_numericalize(vocab, tokens):
    #remove_constants={}
    remove_constants={
        Constants.PAD, Constants.START, Constants.END}
          
    sentences = []
    for ex in tokens:
        end, words_list = False, []
        for x in ex:
            word = vocab.itos[x]
            end = True if word == Constants.END else end
            if word not in remove_constants and not end:
                words_list.append(word)
            else:
                pass
        sentences.append(words_list)

    return sentences