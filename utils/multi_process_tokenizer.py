import nltk
import multiprocessing

#multiprocessing not completed
def tokenizer(strings:'list of string',type:str='word',join:bool=True)->'list of results':
    assert type=='word' or type=='sen'
    results=[]
    if type=='word':
        for s in strings:
            if join:
                results.append(' '.join(nltk.word_tokenize(s)))
            else:
                results.append(nltk.word_tokenize(s))
    else:
        for s in strings:
            if join:
                results.append(' '.join(nltk.sent_tokenize(s)))
            else:
                results.append(nltk.sent_tokenize(s))
    return results



