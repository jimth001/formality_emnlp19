import nltk

def load_fasttext_word_embedding(path):
    vectors = []
    vocab_hash = {}
    with open(path, 'r', encoding='utf-8') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            strs = line.strip().split(' ')
            vocab_hash[strs[0]] = len(vectors)
            vectors.append([float(s) for s in strs[1:]])
    return vectors, vocab_hash

def load_corpus_and_stat_vocab(path):
    freq_vocab={}
    corpus=[]
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            sens=break_sentence(line,skip=True)
            new_sens=[]
            for s in sens:
                words=tokenizer(s,join=False,only_split=False)
                for w in words:
                    if w in freq_vocab:
                        freq_vocab[w]+=1
                    else:
                        freq_vocab[w]=1
                new_sens.append(' '.join(words))
            corpus.append(' '.join(new_sens))
    return corpus,freq_vocab




def break_sentence(paragraph,skip=False,punctuations=None):
    if skip:
        return [paragraph]
    if punctuations is None:
        punctuations = ['?', '？', '...', '......', '!','.',',','，']
    sens=[]
    sens.append(paragraph)
    new_sens = []
    one_sen = ''
    for char in paragraph:
        if char in punctuations:
            if one_sen!='':
                one_sen += (' ' + char)
            else:
                one_sen+=char
            new_sens.append(one_sen)
            one_sen = ''
        else:
            one_sen+=char
    if one_sen!='':
        new_sens.append(one_sen)
    return new_sens

def tokenizer(sentence,join=False,only_split=True):
    if only_split:
        if join:
            return sentence
        else:
            return sentence.split()
    else:
        if join:
            return ' '.join(nltk.word_tokenize(sentence))
        else:
            return nltk.word_tokenize(sentence)

def break_sen_and_tokernize(para,break_sen=False):
    if break_sen:
        return nltk.word_tokenize(' '.join(break_sentence(para)))
    else:
        return nltk.word_tokenize(para)

if __name__=='__main__':
    #print(' a a '.strip())
    #print('Do not belive in it at all.'.split('?'))
    a=break_sentence('This question is for girls,"Have you ever gone out with a guy that you liked, but you did not really know?".')
    print(a)
    print(' '.join(a))
    #print(nltk.word_tokenize(' '.join(a)))