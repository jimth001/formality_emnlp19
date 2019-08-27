def load_word_embedding(path,tool='fasttext'):
    if tool=='fasttext':
        vectors=[]
        vocab_hash={}
        with open(path,'r',encoding='utf-8') as f:
            first_line=True
            for line in f:
                if first_line:
                    first_line=False
                    continue
                strs=line.strip().split(' ')
                vocab_hash[strs[0]]=len(vectors)
                vectors.append([float(s) for s in strs[1:]])
        return vectors,vocab_hash
    else:
        return (None,None)