import nltk

def get_n_gram_list(tokens,n_grams):
    a_list=[]
    len_tokens=len(tokens)
    for i in range(0, n_grams):
        for j in range(0,len_tokens-i):
            a_list.append(' '.join(tokens[j:j+i+1]))
    return a_list

def cal_pinc_for_one_pair(src_tokens,gen_tokens,n_grams):
    src_n_gram_list=get_n_gram_list(src_tokens,n_grams=n_grams)
    gen_n_gram_list=get_n_gram_list(gen_tokens,n_grams=n_grams)
    counter=0
    for item in gen_n_gram_list:
        if item in src_n_gram_list:
            counter+=1
    if len(gen_n_gram_list)==0:
        return 0
    return 1-counter/len(gen_n_gram_list)

def load_file_and_tokenize(file):
    sens=[]
    with open(file,'r',encoding='utf-8') as f:
        for line in f:
            sens.append(nltk.word_tokenize(line.strip()))
    return sens

def cal_file_pinc(src_file,gen_file,n_grams):
    src_sens=load_file_and_tokenize(src_file)
    gen_sens=load_file_and_tokenize(gen_file)
    score=0
    assert len(src_sens)==len(gen_sens)
    ind=[i for i in range(0,len(src_sens))]
    for s,g,i in zip(src_sens,gen_sens,ind):
        score+=cal_pinc_for_one_pair(s,g,n_grams=n_grams)
    return score/len(src_sens)

def evaluate_pinc(resources):
    def eval_factory(log_dict,re):
        src_file=re['input']
        for key in re:
            log_dict[key]=cal_file_pinc(src_file,re[key],n_grams=4)
    eval_log={}
    for key in resources.keys():
        eval_log[key]={}
        eval_factory(eval_log[key], resources[key])
    return eval_log


