import nltk

def file_tokenize(input,output):
    with open(input,'r',encoding='utf-8') as f:
        with open(output,'w',encoding='utf-8') as fw:
            for line in f:
                fw.write(' '.join(nltk.word_tokenize(line.strip()))+'\n')

