from gpt.src import encoder

def read_file_lines(path):
    lines=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def write_file_lines(path,lines,tokenizer=None,max_len=150):
    data_dropped=False
    with open(path,'w',encoding='utf-8') as fw:
        for line in lines:
            if tokenizer is None:
                fw.write(line.strip()+'\n')
            else:
                if len(tokenizer.encode(line.strip()))<max_len:
                    fw.write(line.strip() + '\n')
                else:
                    #print(line)
                    data_dropped=True
    return data_dropped
