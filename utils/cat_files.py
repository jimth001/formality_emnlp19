from utils.file_api import read_file_lines,write_file_lines

def cat_files(f_list,out_path,tokenizer,max_len):
    f_lines=[read_file_lines(f) for f in f_list]
    new_lines=[]
    f_len=[len(x) for x in f_lines]
    print(f_len)
    assert max(f_len)==min(f_len)
    length=max(f_len)
    for i in range(0,length):
        texts=[]
        for j in range(0,len(f_lines)):
            texts.append(f_lines[j][i])
        new_lines.append('\t'.join(texts))
    return write_file_lines(path=out_path,lines=new_lines,tokenizer=tokenizer,max_len=max_len)

