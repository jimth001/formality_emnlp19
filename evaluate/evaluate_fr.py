import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from evaluate.bleu.nltk_bleu import evaluate_bleu
from evaluate.formality.classifier_fr import evaluate_formality
from evaluate.PINC.pinc import evaluate_pinc
def get_default_resources(domain='fr',to_fm=True,to_inf=False):
    def factory(file_dict,in_dir,out_dir,ref_dir,in_flag='informal',target_flag='formal'):
        '''file_dict['rule_based']=out_dir+target_flag+'.rule_based'
        file_dict['pbmt']=out_dir+target_flag+'.pbmt'
        file_dict['nmt_baseline']=out_dir+target_flag+'.nmt_baseline'
        file_dict['nmt_copy']=out_dir+target_flag+'.nmt_copy'
        file_dict['nmt_combined']=out_dir+target_flag+'.nmt_combined'''
        file_dict['input']=in_dir+in_flag
        file_dict['ref0']=ref_dir+target_flag+'.ref0'
        file_dict['ref1'] = ref_dir+target_flag+'.ref1'
        file_dict['ref2'] = ref_dir+target_flag+'.ref2'
        file_dict['ref3'] = ref_dir+target_flag+'.ref3'
        file_dict['gpt_rule'] = out_dir + target_flag + '.gpt.rule'
        file_dict['gpt_ori'] = out_dir + target_flag + '.gpt.ori'
        file_dict['gpt.hie.ori_rule'] = out_dir + target_flag + '.gpt.hieori_rule'
        file_dict['gpt.cat_no_share.ori_rule'] = out_dir + target_flag + '.gpt.cat_no_share.ori_rule'
        file_dict['gpt.ori_rule.ens'] = out_dir + target_flag + '.gpt.ori_rule.ens'
        #file_dict['FT']=out_dir+target_flag+'.MultiTask-tag-style'
        file_dict['gpt.cat.domain_cmb.ori_rule'] = out_dir + target_flag + '.gpt.cat.domain_cmb.ori_rule'
    resources={}
    data_path='./gyafc_model_outputs/'
    if to_fm:
        resources['inf2fm']={}
        factory(resources['inf2fm'], data_path + domain + '_in/',
                data_path + domain + '_out/', data_path + domain + '_refs/',
                in_flag='informal', target_flag='formal')
    if to_inf:
        resources['fm2inf']={}
        factory(resources['fm2inf'], data_path + domain + '_in/',
                data_path + domain + '_out/', data_path + domain + '_refs/',
                in_flag='formal', target_flag='informal')
    return resources

def print_dict(d):
    for key in d:
        print(key)
        nd=d[key]
        for k in nd:
            print(k,nd[k])

if __name__=='__main__':
    re=get_default_resources(domain='fr')
    bleu_result=evaluate_bleu(re)
    pinc_result=evaluate_pinc(re)
    print_dict(bleu_result)
    print_dict(pinc_result)