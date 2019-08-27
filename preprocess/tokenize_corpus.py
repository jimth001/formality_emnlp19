from utils.multi_process_tokenizer import tokenizer
from utils.file_api import read_file_lines,write_file_lines
domains = ['Entertainment_Music', 'Family_Relationships']


def tok():
    ori_path = '../training_data/'
    in_files = []
    out_files = []
    for domain in domains:
        in_files.append(ori_path + 'ori/' + domain + '/train/formal')
        in_files.append(ori_path + 'ori/' + domain + '/train/informal')
        in_files.append(ori_path + 'ori/' + domain + '/tune/formal.ref0')
        in_files.append(ori_path + 'ori/' + domain + '/tune/informal')
        in_files.append(ori_path + 'ori/' + domain + '/test/formal.ref0')
        in_files.append(ori_path + 'ori/' + domain + '/test/informal')
        in_files.append(ori_path + 'ori/' + domain + '/tune/formal')
        in_files.append(ori_path + 'ori/' + domain + '/test/formal')
        out_files.append(ori_path + 'ori/' + domain + '/train/formal.tok')
        out_files.append(ori_path + 'ori/' + domain + '/train/informal.tok')
        out_files.append(ori_path + 'ori/' + domain + '/tune/formal.ref0.tok')
        out_files.append(ori_path + 'ori/' + domain + '/tune/informal.tok')
        out_files.append(ori_path + 'ori/' + domain + '/test/formal.ref0.tok')
        out_files.append(ori_path + 'ori/' + domain + '/test/informal.tok')
        out_files.append(ori_path + 'ori/' + domain + '/tune/formal.tok')
        out_files.append(ori_path + 'ori/' + domain + '/test/formal.tok')
    for f_in,f_out in zip(in_files,out_files):
        sens=read_file_lines(f_in)
        s_tok=tokenizer(sens,type='word',join=True)
        write_file_lines(path=f_out,lines=s_tok)

if __name__=='__main__':
    tok()
    print("all work has finished")



