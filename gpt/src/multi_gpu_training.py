import tensorflow as tf
from gpt.src.gpt2 import GPT2
import tensorflow.contrib.slim as slim
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np
from datetime import timedelta
import time
import os

class multi_gpu_trainer:
    def __init__(self,device_id:list,model_fn:GPT2):
        self.device_id=device_id
        self.model_fn=model_fn
        self.tower_grads = []
        self.accum_vars={}
        self.accum_grad_ops={}
        self.zero_ops={}
        self.learning_rate=1e-4
        self.input={}
        self.input_len={}
        self.target={}
        self.target_mask={}
        self.graph=tf.Graph()
        self.vars_for_infer = []
        self.vars_for_train = []
        self.losses=[]
        self.only_predict_target=True
        self.sep_flag = '\t'
        self.sep_num = 2
        self.replaced_flag = '\t'
        tf.logging.set_verbosity(tf.logging.INFO)


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


    def build_data_parallel_training_graph(self):
        with self.graph.as_default():
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.accum_vars={}
            self.zero_ops={}
            self.accum_grad_ops={}
            for i in range(0, len(self.device_id)):
                with tf.variable_scope(tf.get_variable_scope(), reuse=(i != 0)):
                    with tf.device('/gpu:%d' % self.device_id[i]):
                        with tf.name_scope('parallel_%d' % i) as scope:
                            self.input[i] = tf.placeholder(tf.int32, [None, None], name='input_%d' % i)
                            self.input_len[i] = tf.placeholder(tf.int32, [None, ], name='input_len_%d' % i)
                            self.target[i] = tf.placeholder(tf.int32, [None, None], name='target_%d' % i)
                            if self.only_predict_target:
                                self.target_mask[i] = tf.placeholder(tf.float32, [None, None], name='mask_%d' % i)
                            else:
                                self.target_mask[i] = None
                            loss = self.model_fn.build_training_graph(self.input[i], self.input_len[i], self.target[i], self.target_mask[i])
                            self.losses.append(loss)
                            grads = self.opt.compute_gradients(loss)
                            tvs = tf.trainable_variables()
                            self.accum_vars[i] = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                                                  for tv in
                                                  tvs]
                            self.zero_ops[i] = [tv.assign(tf.zeros_like(tv)) for tv in self.accum_vars[i]]
                            self.accum_grad_ops[i] = [self.accum_vars[i][j].assign_add(gv[0]) for j, gv in
                                                      enumerate(grads)]
                            self.tower_grads.append([(self.accum_vars[i][j], gv[1]) for j, gv in enumerate(grads)])
            grads = self.average_gradients(self.tower_grads)
            with tf.device('/gpu:0'):
                self.accum_steps=tf.placeholder(tf.float32, [], name='accum_stpes')
                self.train_step = self.opt.apply_gradients([(g/self.accum_steps, v) for g,v in grads])
                self.avg_loss=tf.stack(self.losses,axis=0)
                self.avg_loss=tf.reduce_mean(self.avg_loss)


    def create_session_init_and_print_all_trainable_vars(self, max_to_save):
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
            return sess

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


    def train_or_eval_batch_with_raw_text(self, sess, input_text, mini_batch, eos_id, append_eos=False, is_train=True,
                                          run_options=None,):
        #batchsize=minibatch*devicenum*k
        device_num=len(self.device_id)
        ori_mini_batch=mini_batch
        mini_batch=mini_batch*device_num
        batch_size = len(input_text)
        batch_input = []
        batch_target = []
        batch_input_len = []
        batch_mask = []
        sep_flag_id=self.model_fn.text_enc.encode(self.sep_flag)[0]
        rep_flag_id=self.model_fn.text_enc.encode(self.replaced_flag)[0]
        for text in input_text:
            input_tokens = self.model_fn.text_enc.encode(text)
            if self.only_predict_target:
                sep = 0
                mask = []
                for i,token in enumerate(input_tokens):
                    if token == sep_flag_id:
                        sep += 1
                        if sep==self.sep_num:
                            input_tokens[i]=rep_flag_id
                    if sep >= self.sep_num:
                        mask.append(1.0)
                    else:
                        mask.append(0.0)
                if append_eos:
                    batch_mask.append(mask)
                else:
                    batch_mask.append(mask[:-1])
            if append_eos:
                batch_input_len.append(len(input_tokens))
                batch_input.append(input_tokens)
                batch_target.append(input_tokens[1:] + [eos_id])
            else:
                batch_input_len.append(len(input_tokens) - 1)
                batch_input.append(input_tokens[:-1])
                batch_target.append(input_tokens[1:])
        # gradient accum and update
        with self.graph.as_default():
            data_num = batch_size
            losses = []
            low = 0
            if is_train:
                sess.run([self.zero_ops[i] for i in range(0,device_num)])
            while low < data_num:
                n_samples = min([mini_batch, data_num - low])
                mini_batch_input = batch_input[low:low + n_samples]
                mini_batch_target = batch_target[low:low + n_samples]
                mini_batch_input_len = batch_input_len[low:low + n_samples]
                mini_batch_target_mask = batch_mask[low:low + n_samples]
                mini_batch_input_padded, _ = self.padding_batch(mini_batch_input)
                mini_batch_target_padded, _ = self.padding_batch(mini_batch_target)
                if self.only_predict_target:
                    self.padding_for_target_mask(mini_batch_target_mask,mini_batch_input_len)
                feed_dict={}
                for i in range(0,device_num):
                    feed_dict[self.input[i]]=mini_batch_input_padded[i*ori_mini_batch:(i+1)*ori_mini_batch]
                    feed_dict[self.target[i]] = mini_batch_target_padded[i * ori_mini_batch:(i + 1) * ori_mini_batch]
                    feed_dict[self.input_len[i]] = mini_batch_input_len[i * ori_mini_batch:(i + 1) * ori_mini_batch]
                    if self.only_predict_target:
                        feed_dict[self.target_mask[i]]= mini_batch_target_mask[i * ori_mini_batch:(i + 1) * ori_mini_batch]
                if is_train:
                    result = sess.run([self.accum_grad_ops[i] for i in range(0,device_num)]+[self.avg_loss], feed_dict=feed_dict, options=run_options)
                    loss=result[-1]
                else:
                    loss = sess.run(self.avg_loss, feed_dict=feed_dict)
                low += n_samples
                losses.append(loss*n_samples)
            if is_train:
                sess.run(self.train_step,feed_dict={self.accum_steps:batch_size/(device_num*ori_mini_batch)})
        return sum(losses) / batch_size

    def padding_batch(self, input_list):
        in_len = [len(i) for i in input_list]
        new_in = pad_sequences(input_list, padding='post')
        return new_in, in_len


    def training(self, eos_id, train_corpus='./story/story.train', dev_corpus='./story/story.dev',
                 init_step_num=1, learning_rate=1e-4, batch_size=64, mini_batch=16, total_steps=100000,
                 train_ckpt_path='./models/117M/model_train_1/', infer_ckpt_path='./models/117M/',
                 eval_per_n_steps=1, max_to_save=3, early_stop_steps=6000,append_eos=True):
        device_num=len(self.device_id)
        assert batch_size%device_num==0
        assert batch_size>mini_batch*device_num and batch_size%(mini_batch*device_num)==0
        self.learning_rate=learning_rate
        sess=self.create_session_init_and_print_all_trainable_vars(max_to_save)
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
                                                                eos_id=eos_id,
                                                                run_options=run_options,
                                                                mini_batch=mini_batch,
                                                                append_eos=append_eos)
            ###eval:
            if step % eval_per_n_steps == 0:
                eval_low = 0
                eval_losses = []
                while eval_low < eval_data_num:
                    eval_n_samples = min([batch_size, eval_data_num - eval_low])
                    eval_losses.append(self.train_or_eval_batch_with_raw_text(
                        sess, dev[eval_low:eval_low + eval_n_samples], eos_id=eos_id, is_train=False, mini_batch=mini_batch,append_eos=append_eos))
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
        print('all work has finished')


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



