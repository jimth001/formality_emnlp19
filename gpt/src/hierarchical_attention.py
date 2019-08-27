import json
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
from gpt.src import model
from gpt.src import beamsearch
import tensorflow.contrib.slim as slim
from datetime import timedelta
import time
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from gpt.src.single_gpu_serving import beam_search_generator
from utils.file_api import read_file_lines,write_file_lines
from gpt.src.model import positions_for,Encoder,Decoder
from utils.cat_files import cat_files
from gpt.config import *


class NMT_GPT():
    def __init__(self,input_num,config_path):
        self.hparams = model.default_hparams()
        self.config_path = config_path
        with open(os.path.join(self.config_path, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))
        self.input_num=input_num
        self.text_enc = encoder.get_encoder(self.config_path)
        self.sos_id=self.text_enc.encode('\t')[0]
        self.eos_id=self.text_enc.encode('\n')[0]

    def def_placeholder_and_components(self):
        # embeddings:
        with tf.variable_scope('encoder'):
            with tf.variable_scope('model'):
                self.wpe = tf.get_variable('wpe', [self.hparams.n_ctx, self.hparams.n_embd],
                                   initializer=tf.random_normal_initializer(stddev=0.01))
                self.wte = tf.get_variable('wte', [self.hparams.n_vocab, self.hparams.n_embd],
                                   initializer=tf.random_normal_initializer(stddev=0.02))
        self.encoder = Encoder('encoder', self.hparams)
        self.decoder = Decoder('encoder', self.hparams)
        self.inputs = [tf.placeholder(tf.int32, [None, None], name='input_%d' % i) for i in range(0, self.input_num)]
        self.input_lens = [tf.placeholder(tf.int32, [None, ], name='input_len_%d' % i) for i in
                           range(0, self.input_num)]
        self.target_in = tf.placeholder(tf.int32, [None, None], name='target_in')
        self.target_out = tf.placeholder(tf.int32, [None, None], name='target_out')
        self.target_len = tf.placeholder(tf.int32, [None], name='target_len')



    def build_training_model(self):
        self.def_placeholder_and_components()
        emb_out=[]
        enc_h_out=[]
        past_for_decoder=[]
        for i in range(0,self.input_num):
            past_length=0
            h = tf.gather(self.wte, self.inputs[i]) + tf.gather(self.wpe, positions_for(self.inputs[i], past_length))
            emb_out.append(h)
            presents, h_enc=self.encoder.encode(h,self.input_lens[i])
            enc_h_out.append(h_enc)
            past_for_decoder.append(presents)
        all_logits=self.decoder.decode_all(tokens=self.target_in,past_list=past_for_decoder,enc_h_list=enc_h_out)['logits']
        with tf.name_scope('loss'):
            batch_max_seq_len = tf.shape(self.target_in)[1]
            target_mask = tf.sequence_mask(self.target_len, maxlen=batch_max_seq_len, dtype=tf.float32)
        cost = sequence_loss(logits=all_logits, targets=self.target_out,
                             weights=target_mask)
        return cost


    def build_beam_search_graph(self, beam_size, batch_size, max_decode_length, decode_alpha=0.6):
        self.def_placeholder_and_components()
        emb_out = []
        enc_h_out = []
        past_for_decoder = []
        for i in range(0, self.input_num):
            past_length = 0
            h = tf.gather(self.wte, self.inputs[i]) + tf.gather(self.wpe, positions_for(self.inputs[i], past_length))
            emb_out.append(h)
            presents, h_enc = self.encoder.encode(h, self.input_lens[i])
            enc_h_out.append(h_enc)
            past_for_decoder.append(presents)
        past_length = 0 if enc_h_out[0] is None else tf.shape(enc_h_out[0])[-2]
        self.decoder.sef_var_for_beam_search(past_length,enc_h_out,beam_size=beam_size)
        with tf.name_scope('beam_search'):
            init_seq = tf.fill(dims=(batch_size, 1), value=self.sos_id)
            seqs, scores = beamsearch.create_inference_graph(init_seqs=init_seq, state=past_for_decoder,
                                                             step_fn=self.decoder.decode_one_step, hparams=self.hparams,
                                                             decode_length=max_decode_length,
                                                             batch_size=batch_size, beam_size=beam_size,
                                                             decode_alpha=decode_alpha, eos_id=self.eos_id,
                                                             ensemble=False, concat_state_dim=None)
        return seqs, scores


class NMT_GPT_Trainer():
    def __init__(self,model_fn:NMT_GPT):
        self.model_fn=model_fn
        self.learning_rate=1e-4
        self.sep_flag='\t'
        self.graph=tf.Graph()
        self.vars_for_infer = []
        self.vars_for_train = []
        self.losses=[]
        self.only_predict_target=True
        tf.logging.set_verbosity(tf.logging.INFO)
        self.is_hierarchical=True
        self.hier_enc_end_token=self.model_fn.text_enc.encode('\t')


    def average_gradients(self,tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
            grads = tf.concat(grads, 0)
            grad = tf.reduce_mean(grads, 0)
            grad_and_var = (grad, grad_and_vars[0][1])
            # [(grad0, var0),(grad1, var1),...]
            average_grads.append(grad_and_var)
        return average_grads


    def build_graph(self):
        with self.graph.as_default():
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.tower_grads = []
            loss = self.model_fn.build_training_model()
            self.losses.append(loss)
            grads = self.opt.compute_gradients(loss)
            tvs = tf.trainable_variables()
            self.accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                                  for tv in
                                  tvs]
            self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars]
            self.accum_grad_ops = [self.accum_vars[j].assign_add(gv[0]) for j, gv in
                                      enumerate(grads) if gv[0] is not None]
            self.tower_grads.append([(self.accum_vars[j], gv[1]) for j, gv in enumerate(grads) ])
            grads = self.average_gradients(self.tower_grads)
            with tf.device('/gpu:0'):
                self.accum_steps=tf.placeholder(tf.float32, [], name='accum_stpes')
                self.train_step = self.opt.apply_gradients([(g/self.accum_steps, v) for g,v in grads])
                self.avg_loss=tf.stack(self.losses,axis=0)
                self.avg_loss=tf.reduce_mean(self.avg_loss)


    def create_session_init_and_print_all_trainable_vars(self, max_to_save, ori_gpt_model_path=None):
        # Print parameters
        with self.graph.as_default():
            all_weights = {v.name: v for v in tf.trainable_variables()}
            total_size = 0
            for v_name in sorted(list(all_weights)):
                v = all_weights[v_name]
                tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                                str(v.shape).ljust(20))
                v_size = np.prod(np.array(v.shape.as_list())).tolist()
                total_size += v_size
            tf.logging.info("Total trainable variables size: %d", total_size)
            all_var_list = slim.get_variables_to_restore()
            for v in all_var_list:
                if 'Adam' in v.name:
                    self.vars_for_train.append(v)
                elif v.name.startswith('beta'):
                    self.vars_for_train.append(v)
                elif v.name.startswith('parallel'):
                    pass
                elif v.name.startswith('Variable'):
                    pass
                else:
                    self.vars_for_infer.append(v)
            if len(self.vars_for_infer) > 0:
                self.saver_infer = tf.train.Saver(self.vars_for_infer, max_to_keep=max_to_save)
            if len(self.vars_for_train) > 0:
                self.saver_train = tf.train.Saver(self.vars_for_train, max_to_keep=max_to_save)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=self.graph, config=config)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            restore_ops=[]
            if ori_gpt_model_path is not None:
                ckpt = tf.train.latest_checkpoint(ori_gpt_model_path)
                tf.logging.info("Loading %s" % ckpt)
                var_list = tf.train.list_variables(ckpt)
                values = {}
                reader = tf.train.load_checkpoint(ckpt)
                for (name, shape) in var_list:
                    if not name.startswith('model/'):  # ignore global_step
                        continue
                    tensor = reader.get_tensor(name)
                    values[name] = tensor
                for v in self.vars_for_infer:
                    #print(v.name)
                    tmp = '/'.join(v.name.split('/')[1:])
                    v_name = tmp.split(':')[0]
                    if v_name!='model/sen_attn_w':
                        op = tf.assign(v, values[v_name])
                    restore_ops.append(op)
                sess.run(restore_ops)
            return sess


    def padding_batch(self, input_list):
        in_len = [len(i) for i in input_list]
        new_in = pad_sequences(input_list, padding='post')
        return new_in, in_len


    def train_or_eval_batch_with_raw_text(self, sess, input_text, mini_batch, is_train=True,
                                          run_options=None):
        batch_size = len(input_text)
        batch_input = {}
        batch_target_in = []
        batch_target_out =[]
        batch_target_len =[]
        batch_input_len = {}
        for text in input_text:
            strs=text.split(self.sep_flag)
            inputs=strs[:-1]
            target=strs[-1]
            if self.is_hierarchical:
                inputs_tokens = [self.model_fn.text_enc.encode(item)+self.hier_enc_end_token for item in inputs]
            else:
                inputs_tokens = [self.model_fn.text_enc.encode(item) for item in inputs]
            target_tokens=self.model_fn.text_enc.encode(target)
            for i in range(0,len(inputs_tokens)):
                if i not in batch_input:
                    batch_input[i]=[]
                batch_input[i].append(inputs_tokens[i])
                if i not in batch_input_len:
                    batch_input_len[i]=[len(inputs_tokens[i])]
                else:
                    batch_input_len[i].append(len(inputs_tokens[i]))
            tar_in=[self.model_fn.sos_id]+target_tokens
            tar_out=target_tokens+[self.model_fn.eos_id]
            batch_target_len.append(len(tar_out))
            batch_target_in.append(tar_in)
            batch_target_out.append(tar_out)
        # gradient accum and update
        #assert batch_size%mini_batch==0
        with self.graph.as_default():
            data_num = batch_size
            losses = []
            low = 0
            if is_train:
                sess.run(self.zero_ops)
            while low < data_num:
                n_samples = min([mini_batch, data_num - low])
                mini_batch_input = [batch_input[i][low:low + n_samples] for i in range(0,len(batch_input))]
                mini_batch_input_len = [batch_input_len[i][low:low + n_samples] for i in range(0, len(batch_input))]
                mini_batch_target_in = batch_target_in[low:low + n_samples]
                mini_batch_target_out = batch_target_out[low:low + n_samples]
                mini_batch_target_len = batch_target_len[low:low + n_samples]
                mini_batch_target_in_padded, _ = self.padding_batch(mini_batch_target_in)
                mini_batch_target_out_padded, _ = self.padding_batch(mini_batch_target_out)
                feed_dict={}
                for i in range(0,self.model_fn.input_num):
                    p,_ = self.padding_batch(mini_batch_input[i])
                    feed_dict[self.model_fn.inputs[i]]=p
                    feed_dict[self.model_fn.input_lens[i]]=mini_batch_input_len[i]
                feed_dict[self.model_fn.target_in] = mini_batch_target_in_padded
                feed_dict[self.model_fn.target_out] = mini_batch_target_out_padded
                feed_dict[self.model_fn.target_len] = mini_batch_target_len
                if is_train:
                    result = sess.run([self.accum_grad_ops, self.avg_loss], feed_dict=feed_dict, options=run_options)
                    loss=result[-1]
                else:
                    loss = sess.run(self.avg_loss, feed_dict=feed_dict)
                low += n_samples
                losses.append(loss*n_samples)
            if is_train:
                sess.run(self.train_step,feed_dict={self.accum_steps:batch_size/mini_batch})
        return sum(losses) / batch_size


    def training(self, eos_id=None, train_corpus='./story/story.train', dev_corpus='./story/story.dev',
                 init_step_num=1, learning_rate=1e-4, batch_size=64, mini_batch=16, total_steps=100000,
                 train_ckpt_path='./models/117M/model_train_1/', infer_ckpt_path='./models/117M/',
                 eval_per_n_steps=1, max_to_save=3, early_stop_steps=6000,append_eos=True,ori_gpt_model_path=None):
        self.learning_rate=learning_rate
        sess=self.create_session_init_and_print_all_trainable_vars(max_to_save,ori_gpt_model_path=ori_gpt_model_path)
        if ori_gpt_model_path is None:
            self.restore_model_and_init(sess, infer_ckpt_path, train_ckpt_path)
        train = load_corpus(train_corpus)
        # train=[' '.join(['you' for j in range(0,512)]) for i in range(0,512)]
        dev = load_corpus(dev_corpus)
        step = init_step_num
        low = 0
        epoch_num = 1
        train_data_num = len(train)
        eval_data_num = len(dev)
        last_improvement_step = init_step_num
        best_loss = 100000
        saved_steps = []
        tf.logging.info('start training...')
        self.graph.finalize()
        start_time = time.time()
        while step < total_steps:
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            n_samples = min([batch_size, train_data_num - low])
            train_loss = self.train_or_eval_batch_with_raw_text(sess, train[low:low + n_samples],
                                                                run_options=run_options,
                                                                mini_batch=mini_batch,
                                                                )
            ###eval:
            if step % eval_per_n_steps == 0:
                eval_low = 0
                eval_losses = []
                while eval_low < eval_data_num:
                    eval_n_samples = min([batch_size, eval_data_num - eval_low])
                    eval_losses.append(self.train_or_eval_batch_with_raw_text(
                        sess, dev[eval_low:eval_low + eval_n_samples], is_train=False, mini_batch=mini_batch))
                    eval_low += eval_n_samples
                eval_avg_loss = sum(eval_losses) / len(eval_losses)
                time_dif = get_time_dif(start_time)
                if eval_avg_loss < best_loss:
                    best_loss = eval_avg_loss
                    last_improvement_step = step
                    tf.logging.info('save step %d', last_improvement_step)
                    self.save_model(sess, infer_ckpt_path, train_ckpt_path, step=step)
                    saved_steps.append(last_improvement_step)
                    tf.logging.info("%s: step %d: train loss %f; eval loss %f *", time_dif, step, train_loss,
                                    eval_avg_loss)
                    if len(saved_steps) > max_to_save:
                        saved_steps = saved_steps[1:]
                else:
                    tf.logging.info("%s: step %d: train loss %f; eval loss %f", time_dif, step, train_loss,
                                    eval_avg_loss)
                    if step - last_improvement_step > early_stop_steps:
                        tf.logging.info("early stopping...")
                        break
            ###
            step += 1
            low += n_samples
            if low == train_data_num:
                low = 0
                epoch_num += 1
        sess.close()
        print('all work has finished')


    def restore_model_and_init(self, sess, ckpt_for_infer, ckpt_for_train):
        with self.graph.as_default():
            if ckpt_for_infer is not None:
                ckpt = tf.train.latest_checkpoint(ckpt_for_infer)
                if ckpt is not None:
                    self.saver_infer.restore(sess, ckpt)
                    tf.logging.info('restored inferring params from %s',ckpt)
            if ckpt_for_train is not None:
                ckpt = tf.train.latest_checkpoint(ckpt_for_train)
                if ckpt is not None:
                    self.saver_train.restore(sess, ckpt)
                    tf.logging.info('restored training params from %s', ckpt)


    def save_model(self, sess, infer_ckpt_path, train_ckpt_path, step):
        with self.graph.as_default():
            if infer_ckpt_path is not None and len(self.vars_for_infer) > 0:
                self.saver_infer.save(sess, os.path.join(infer_ckpt_path,'model'), global_step=step)
            if train_ckpt_path is not None and len(self.vars_for_train) > 0:
                self.saver_train.save(sess, os.path.join(train_ckpt_path,'model'), global_step=step)


    def padding_for_target_mask(self,mask_list,input_len):
        batch_size= len(mask_list)
        assert batch_size==len(input_len)
        max_len=max(input_len)
        for i in range(0,batch_size):
            l=input_len[i]
            mask_list[i]=mask_list[i]+[0.0]*(max_len-l)



def load_corpus(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def test(config_path,input_num,model_dir='./models/ori_rule/formality_infer/',input_path='../training_data/dif_models/eval.ori_rule',
         output_path='../evaluate/gyafc_model_outputs/fr_out/formal.gpt.cat_ori_rule.old',beam_size=4,max_dec_len=60,dec_alpha=0.6):
    gpt2 = NMT_GPT(config_path=config_path,input_num=input_num)
    generator = beam_search_generator(gpt2, beam_size=beam_size,
                                      model_directory=model_dir, max_dec_len=max_dec_len,
                                      dec_alpha=dec_alpha)
    sess=generator.build_graph_and_restore(eos_id=gpt2.text_enc.encode('\n')[0])
    lines=read_file_lines(input_path)
    result=[]
    for line in lines:
        result.append(generator.generate(sess,line,multi_pls=True))
        print(line+' ||| '+result[-1].strip())
    sess.close()
    write_file_lines(output_path, result)


def train(config_path,input_num,ori_gpt_model=None,sep_flag='\t',
          train_corpus='../training_data/preprocessed/Family_Relationships/train.ori.txt',
          dev_corpus='../training_data/preprocessed/Family_Relationships/val.ori.txt',
          infer_ckpt_path='./models/ori_data_fr/formality_infer/',
          train_ckpt_path='./models/ori_data_fr/formality_train/'):
    gpt2 = NMT_GPT(input_num,config_path)
    trainer = NMT_GPT_Trainer(gpt2)
    trainer.build_graph()
    trainer.sep_flag=sep_flag
    trainer.training(train_corpus=train_corpus,
                     dev_corpus=dev_corpus,
                     infer_ckpt_path=infer_ckpt_path, train_ckpt_path=train_ckpt_path,
                     learning_rate=1e-4, init_step_num=1,
                     batch_size=128, mini_batch=16,
                     eval_per_n_steps=100,
                     total_steps=3000,
                     early_stop_steps=200,
                     max_to_save=2,
                     append_eos=True,
                     eos_id=gpt2.text_enc.encode('\n')[0],ori_gpt_model_path=ori_gpt_model)


def HA(domain='fr',max_len_limit=220,only_test=False):
    methods = ['ori', 'rule']
    model_path='./models_hie_'+domain+'/'+'_'.join(methods)
    init_model_path = './models/formality_infer'
    if not os.path.exists('./models_hie_'+domain):
        os.mkdir('./models_hie_'+domain)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'/formality_train')
        shutil.copytree(init_model_path, model_path+'/formality_infer')
    data_path = '../training_data/dif_models_'+domain+'/'
    cat_files([data_path + 'informal.train.'+m for m in methods]+ [ data_path + 'formal.train.rule', ],
              data_path + 'train.'+'_'.join(methods),
              tokenizer=text_enc, max_len=max_len_limit)
    cat_files([data_path + 'informal.val.' + m for m in methods] + [data_path + 'formal.val.rule', ],
              data_path + 'val.' + '_'.join(methods),
              tokenizer=text_enc, max_len=max_len_limit)
    lp = cat_files([data_path + 'informal.test.' + m for m in methods],
                   data_path + 'eval.' + '_'.join(methods),
                   tokenizer=text_enc, max_len=max_len_limit)
    if lp:
        print('_'.join(methods)+' data droped')
    if not only_test:
        train(config_path=config_path,input_num=len(methods),sep_flag='\t', ori_gpt_model=init_model_path,
          train_corpus=data_path + 'train.'+'_'.join(methods),
          dev_corpus=data_path + 'val.'+'_'.join(methods),
          infer_ckpt_path=model_path+'/formality_infer',
          train_ckpt_path=model_path+'/formality_train')
    test(config_path=config_path,input_num=len(methods),
         model_dir=model_path+'/formality_infer',
         input_path=data_path + 'eval.'+'_'.join(methods),
         output_path='../evaluate/gyafc_model_outputs/' + domain + '_out/formal.gpt.hie'+'_'.join(methods))
