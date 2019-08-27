from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk

def bleu(reference_files_src_list,gen_file_src,ngrams=4,ignore_case=False):
    all_reference=[]
    for src in reference_files_src_list:
        with open(src,'r',encoding='utf-8') as f:
            one_reference=[]
            for line in f:
                if not ignore_case:
                    one_reference.append(nltk.word_tokenize(line.strip()))
                else:
                    one_reference.append(nltk.word_tokenize(line.strip().lower()))
            all_reference.append(one_reference)
    all_reference=[[all_reference[i][j] for i in range(0,len(all_reference))] for j in range(0,len(all_reference[0]))]
    gen=[]
    with open(gen_file_src,'r',encoding='utf-8') as f:
        for line in f:
            if not ignore_case:
                gen.append(nltk.word_tokenize(line.strip()))
            else:
                gen.append(nltk.word_tokenize(line.strip().lower()))
    weight=[1.0/ngrams]*ngrams
    print(len(gen))
    b=corpus_bleu(all_reference,gen,weights=weight)
    return b


def get_ref_src_list(path_prefix,ref_num=4):
    src_list=[]
    for i in range(0,ref_num):
        src_list.append(path_prefix+str(i))
    return src_list


def evaluate_bleu(resources):
    def eval_factory(log_dict,re):
        ref_list=[re['ref0'],re['ref1'],re['ref2'],re['ref3']]
        for key in re:
            print(key,len(re[key]))
            log_dict[key]=bleu(ref_list,re[key])
    eval_log={}
    for key in resources.keys():
        eval_log[key]={}
        eval_factory(eval_log[key], resources[key])
    return eval_log