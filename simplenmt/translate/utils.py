from data.constants import Constants


def de_numericalize(vocab, tokens):
    """
    Use vocab to transform tokens to sentences.
    """

   # Some constants don't wish to print, like <sos>, <pad> and <eos> 
    remove_constants={
        Constants.PAD, Constants.START, Constants.END, Constants.UNK}

    sentences = list()
    for row in tokens:
        words_list = list()
        for word_id in row:
            word = vocab.itos[word_id]
            if word == Constants.END:
                break
            if word not in remove_constants:
                words_list.append(word)
        sentences.append(words_list)

    return sentences

"""
def de_numericalize(vocab, tokens):
    # Use vocab to transform tokens to sentences.
    
    # Some constants don't wish to print, like <sos>, <pad> and <eos> 
    remove_constants={
        Constants.PAD, Constants.START, Constants.END, Constants.UNK}

    sentences = list()
    for row in tokens:
        end = False
        words_list = list()
        for word_id in row:
            word = vocab.itos[word_id]
            end = True if word == Constants.END else end
            if word not in remove_constants and not end:
                words_list.append(word)
            else:
                pass
        sentences.append(words_list)

    return sentences
"""