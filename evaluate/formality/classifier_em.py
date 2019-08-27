import os
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer
import time
import pickle
from datetime import timedelta
from utils import embedding_api
import random

data_path='./new_exp_em/classifier/'
class NNModel:
    def __init__(self, embedding, mode, model_path=None, vocab_hash=None):
        self.graph=tf.Graph()
        self.learning_rate=1e-3
        self.batch_size=256
        self.epoch_num=10
        self.dropout_keep_prob=1.0
        self.vocab_hash=vocab_hash
        self.embedding = embedding
        self.vocab_size = embedding.shape[0]
        with self.graph.as_default():
            self.input_x = tf.placeholder(tf.int64, [None, None], name='input_x')
            self.x_sequence_len = tf.placeholder(tf.int64, [None], name='x_sequence_len')
            self.embedding_ph = tf.placeholder(tf.float32, [self.embedding.shape[0], self.embedding.shape[1]],
                                               name='embedding')
            self.input_y = tf.placeholder(tf.int64, [None], name='input_y')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.inputs_data=[self.input_x,self.x_sequence_len]
        self.save_dir='./new_exp_em/classifier/model/'
        self.print_per_batch=100
        self.require_improvement=6400
        if mode=='train':
            pass
        elif mode=='eval' or mode=='predict':
            pass
        else:
            assert False,'no this mode:'+str(mode)


    def __get_time_dif(self, start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def build_basic_rnn_model(self,rnn_unit_num=32,dense_layer_unit_num=8,class_num=2,reg_para=0.0):
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                word_embedding = tf.get_variable(name='embedding', shape=self.embedding.shape, dtype=tf.float32,
                                                 trainable=True)
                self.embedding_init = word_embedding.assign(self.embedding_ph)
                x_embedding = tf.nn.embedding_lookup(word_embedding, self.input_x)
            with tf.name_scope("rnn"):
                fw_cell = rnn.LSTMCell(rnn_unit_num)
                bw_cell = rnn.LSTMCell(rnn_unit_num)
                out, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x_embedding, self.x_sequence_len,
                                                             dtype=tf.float32,
                                                             initial_state_fw=None, initial_state_bw=None)
                # combined_out = tf.concat(out, axis=2)
                combined_state = tf.concat([state[0][1], state[1][1]], axis=1)
            with tf.name_scope("dense_layers"):
                fc = tf.layers.dense(combined_state, dense_layer_unit_num, name='fc1', activation=tf.nn.tanh,
                                     kernel_initializer=xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_para),
                                     bias_initializer=tf.zeros_initializer(),
                                     bias_regularizer=tf.contrib.layers.l2_regularizer(reg_para))
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                self.logits = tf.layers.dense(fc, class_num, name='fc2', activation=tf.nn.tanh,
                                              kernel_initializer=xavier_initializer(),
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_para),
                                              bias_initializer=tf.zeros_initializer(),
                                              bias_regularizer=tf.contrib.layers.l2_regularizer(reg_para))
                self.y_pred_value = tf.nn.softmax(self.logits)
                self.y_pred_class = tf.argmax(self.y_pred_value, 1)
            with tf.name_scope("optimize"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits + 1e-10,
                                                                               labels=self.input_y)
                self.loss = tf.reduce_mean(cross_entropy)
                self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            with tf.name_scope("evaluate_metrics"):
                correct_pred = tf.equal(self.input_y, self.y_pred_class)
                self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _evaluate_without_predict_result(self, input_data, target):
        batches = self.batch_iter([input_data], target, self.batch_size, shuffle=False)
        total_loss = 0.0
        total_acc = 0.0
        total_len = len(target)
        for batch_data, batch_target in batches:
            batch_len = len(batch_target)
            feed_dict = self.feed_data(inputs_data=batch_data, keep_prob=1.0, target=batch_target)
            loss, acc = self.session.run([self.loss, self.acc], feed_dict=feed_dict)
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / total_len, total_acc / total_len


    def predict_prob(self, input_data, model_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            result = []
            with tf.Session(config=config) as sess:
                self.session = sess
                saver.restore(sess, model_path)
                batches = self.batch_iter([input_data], target=None, batch_size=self.batch_size, shuffle=False)
                for batch_data in batches:
                    feed_dict = self.feed_data(inputs_data=batch_data, keep_prob=1.0)
                    pred = self.session.run([self.y_pred_value], feed_dict=feed_dict)[0]
                    for d in pred:
                        result.append(d)
            return result


    def feed_data(self, inputs_data, keep_prob, target=None):
        feed_dict = {}
        for i in range(len(self.inputs_data)):
            feed_dict[self.inputs_data[i]] = inputs_data[i]
        feed_dict[self.keep_prob] = keep_prob
        if not target is None:
            feed_dict[self.input_y] = target
        return feed_dict

    def evaluate(self,input_data,target,model_path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                self.session = sess
                saver.restore(sess, model_path)
                print(self._evaluate_without_predict_result(input_data, target))

    def train_model(self,train_x,train_label,val_x,val_label,continue_train=False,previous_model_path=None):
        start_time = time.time()
        with self.graph.as_default():
            saver = tf.train.Saver(max_to_keep=20)
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            ##############################################################
            print(str(self.__get_time_dif(start_time)) + "trainning and evaluating...")
            total_batch = 0
            best_acc_val = 0.0
            last_improved = 0
            flag = False
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with self.graph.as_default():
                with tf.Session(config=config) as sess:
                    self.session = sess
                    if continue_train is False:
                        sess.run(tf.global_variables_initializer())
                        sess.run(self.embedding_init, feed_dict={self.embedding_ph: self.embedding})
                    else:
                        saver.restore(sess, previous_model_path)
                    self.session.graph.finalize()
                    for epoch in range(self.epoch_num):
                        print("epoch:" + str(epoch + 1))
                        batch_train = self.batch_iter([train_x], train_label, batch_size=self.batch_size, shuffle=False)
                        for batch_data, batch_target in batch_train:
                            feed_dict = self.feed_data(inputs_data=batch_data, target=batch_target,
                                                       keep_prob=self.dropout_keep_prob)
                            s = self.session.run([self.optim], feed_dict=feed_dict)
                            if total_batch == 0:
                                saver.save(sess=self.session, save_path=self.save_dir + "model.ckpt")
                            if total_batch > 0 and total_batch % self.print_per_batch == 0:
                                feed_dict[self.keep_prob] = 1.0
                                loss_train, acc_train = self.session.run([self.loss, self.acc],
                                                                         feed_dict=feed_dict)
                                loss_val, acc_val = self._evaluate_without_predict_result(val_x,
                                                                                          val_label)
                                if acc_val > best_acc_val:
                                    best_acc_val = acc_val
                                    last_improved = total_batch
                                    saver.save(sess=self.session,
                                               save_path=self.save_dir + str(total_batch) + 'model.ckpt')
                                    improved_str = '*'
                                else:
                                    improved_str = ''
                                time_dif = self.__get_time_dif(start_time)
                                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif,
                                                 improved_str))
                            total_batch += 1
                            if total_batch - last_improved > self.require_improvement:
                                print("No optimization for a long time, auto-stopping...")
                                flag = True
                                break
                        if flag:
                            break
                self.session = None


    def batch_iter(self, input_data, target=None, batch_size=64,padding=0,shuffle=False):
        assert not input_data is None,"input_data is None"
        data_len = len(input_data[0])
        num_batch = int((data_len - 1) / batch_size) + 1
        if shuffle:
            indices = np.random.permutation(np.arange(data_len))
        else:
            indices = range(data_len)
        x_shuffle = [input_data[0][i] for i in indices]
        x_seq_len = [len(x_shuffle[i]) for i in range(len(x_shuffle))]
        if target is None:
            for i in range(num_batch):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                batch_x = copy_list(x_shuffle[start_id:end_id])
                batch_x_seq_len = x_seq_len[start_id:end_id]
                x_max_len = max(batch_x_seq_len)
                for list in batch_x:
                    if len(list) < x_max_len:
                        list += [padding] * (x_max_len - len(list))
                yield [batch_x,batch_x_seq_len]
        else:
            y_shuffle = [target[i] for i in indices]
            for i in range(num_batch):
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                batch_y = y_shuffle[start_id:end_id]
                batch_x = copy_list(x_shuffle[start_id:end_id])
                batch_x_seq_len = x_seq_len[start_id:end_id]
                x_max_len = max(batch_x_seq_len)
                for list in batch_x:
                    if len(list) < x_max_len:
                        list += [padding] * (x_max_len - len(list))
                yield [batch_x,batch_x_seq_len],batch_y


def get_file_src_list(parent_path, file_type='.txt'):
    files = os.listdir(parent_path)
    src_list = []
    for file in files:
        absolute_path = os.path.join(parent_path, file)
        if os.path.isdir(absolute_path):
            src_list += get_file_src_list(absolute_path)
        elif file.endswith(file_type):
            src_list.append(absolute_path)
    return src_list


def copy_list(list):
    new_list = []
    for l in list:
        if type(l) == type([0]) or type(l) == type(np.array([0])):
            new_list.append(copy_list(l))
        else:
            new_list.append(l)
    return new_list


class Data:
    def __init__(self,x,y,ori_x=None):
        self.x=x
        self.y=y
        self.ori_x=ori_x
    def split(self):
        self.x=self.x.split(' ')
    def str2index(self,word_dict,with_unk=True):
        index=[]
        if with_unk:
            for s in self.x:
                if s in word_dict:
                    index.append(word_dict[s])
                else:
                    index.append(len(word_dict))
        else:
            for s in self.x:
                if s in word_dict:
                    index.append(word_dict[s])
        self.x=index


def preprocess(informal_src_list,formal_src_list,embedding_path,output_path=None,shuffle=True):
    vectors,vocab_hash=embedding_api.load_word_embedding(embedding_path)
    all_data=[]
    for src in informal_src_list:
        with open(src,'r',encoding='utf-8') as f:
            for line in f:
                d=Data(nltk.word_tokenize(line.strip()), 0, line.strip())
                d.str2index(vocab_hash,with_unk=False)
                all_data.append(d)
    for src in formal_src_list:
        with open(src,'r',encoding='utf-8') as f:
            for line in f:
                d=Data(nltk.word_tokenize(line.strip()), 1, line.strip())
                d.str2index(vocab_hash,with_unk=False)
                all_data.append(d)
    if shuffle:
        random.shuffle(all_data)
    if output_path is not None:
        pickle.dump(all_data,open(output_path,'wb'),protocol=True)
    return all_data


def all_prepro():
    train_inf_src=['./new_exp_em/classifier/informal.train.tok.bpe.len_filtered']
    train_fm_src = ['./new_exp_em/classifier/formal.train.tok.bpe.len_filtered']
    val_inf_src = ['./new_exp_em/classifier/informal.val.tok.bpe']
    val_fm_src = ['./new_exp_em/classifier/formal.val.tok.bpe']
    test_inf_src = ['./new_exp_em/classifier/informal.test.tok.bpe']
    test_fm_src = ['./new_exp_em/classifier/formal.test.tok.bpe']
    embedding_path='./new_exp_em/embedding/embedding.bpe.big.txt'
    preprocess(train_inf_src,train_fm_src,embedding_path=embedding_path,
               output_path='./new_exp_em/classifier/train.pkl')
    preprocess(val_inf_src, val_fm_src, embedding_path=embedding_path,
               output_path='./new_exp_em/classifier/val.pkl')
    preprocess(test_inf_src, test_fm_src, embedding_path=embedding_path,
               output_path='./new_exp_em/classifier/test.pkl')

def use_nn_model():
    train = pickle.load(open('./new_exp_em/classifier/train.pkl', 'rb'))
    val = pickle.load(open('./new_exp_em/classifier/val.pkl', 'rb'))
    embedding_path = './new_exp_em/embedding/embedding.bpe.big.txt'
    embedding, vocab_hash = embedding_api.load_word_embedding(embedding_path)
    nn=NNModel(np.array(embedding),mode='train')
    nn.build_basic_rnn_model()
    nn.train_model([t.x for t in train],[t.y for t in train],[t.x for t in val],[t.y for t in val],
                   continue_train=False, previous_model_path='./new_exp_em/classifier/model/990model.ckpt')

def test():
    test = pickle.load(open('./new_exp_em/classifier/test.pkl', 'rb'))
    embedding_path = './new_exp_em/embedding/corpus.fine_tune_embedding.epoch.10'
    embedding,vocab_hash = embedding_api.load_word_embedding(embedding_path)
    nn = NNModel(np.array(embedding),mode='eval')
    nn.build_basic_rnn_model()
    nn.evaluate([t.x for t in test],[t.y for t in test],model_path='')

def predict(model_path,file_path='./new_exp_em/classifier/val.pkl',embedding_path='./new_exp_em/embedding/corpus.fine_tune_embedding.epoch.10'):
    test = pickle.load(open(file_path, 'rb'))
    embedding, vocab_hash = embedding_api.load_word_embedding(embedding_path)
    nn = NNModel(np.array(embedding),mode='predict')
    nn.batch_size=256
    nn.build_basic_rnn_model()
    result=nn.predict_prob([t.x for t in test], model_path=model_path)
    return test,result


def evaluate_one_formality(input_file_path,is_inf):
    embedding_path = './new_exp_em/embedding/embedding.bpe.big.txt'
    embedding, vocab_hash = embedding_api.load_word_embedding(embedding_path)
    nn = NNModel(np.array(embedding), mode='eval')
    nn.batch_size = 128
    nn.build_basic_rnn_model()
    if is_inf:
        data = preprocess(informal_src_list=[input_file_path], formal_src_list=[], embedding_path=embedding_path,
                          shuffle=False)
    else:
        data = preprocess(informal_src_list=[], formal_src_list=[input_file_path], embedding_path=embedding_path,
                          shuffle=False)
    result = nn.predict_prob([t.x for t in data], model_path='./new_exp_em/classifier/model/600model.ckpt')
    score = 0
    if is_inf:
        for s in result:
            score += s[0]
    else:
        for s in result:
            score += s[1]
    print(score / len(data))
    return score / len(data)

def test_formality_score(files=None):
    if files is None:
        files = {
                 'informal': ['./new_exp_em/classifier/informal.test.tok.bpe'],
                 'formal': ['./new_exp_em/classifier/formal.test.tok.bpe'],
                 'rule_based': ['./data/Entertainment_Music/model_outputs/formal.rule_based.bpe'],
                 'pbmt': ['./data/Entertainment_Music/model_outputs/formal.pbmt.bpe'],
                 'nmt_baseline': ['./data/Entertainment_Music/model_outputs/formal.nmt_baseline.bpe'],
                 'nmt_copy': ['./data/Entertainment_Music/model_outputs/formal.nmt_copy.bpe'],
                 'nmt_combined': ['./data/Entertainment_Music/model_outputs/formal.nmt_combined.bpe'],
                 }
    embedding_path = './new_exp_em/embedding/embedding.bpe.big.txt'
    embedding, vocab_hash = embedding_api.load_word_embedding(embedding_path)
    nn = NNModel(np.array(embedding),mode='eval')
    nn.batch_size = 128
    nn.build_basic_rnn_model()
    eval_log={}
    for key in files.keys():
        if type(files[key])==type([]):
            fm_files=files[key]+'.bpe'
        else:
            fm_files=[files[key]+'.bpe']
        data=preprocess(informal_src_list=[],formal_src_list=fm_files,embedding_path=embedding_path,shuffle=False)
        result = nn.predict_prob([t.x for t in data], model_path='./new_exp_em/classifier/model/600model.ckpt')
        score=0
        for s in result:
            score+=s[1]
        print(key,score/len(data))
        eval_log[key]=score/len(data)
    return eval_log


def cal_formality_score_for_each_sentence(output_dir,files=None):
    if files is None:
        files = {
            'rule_based': ['./data/Family_Relationships/bpe_outputs/formal.rule_based.bpe'],
            'pbmt': ['./data/Family_Relationships/bpe_outputs/formal.pbmt.bpe'],
            'nmt_baseline': ['./data/Family_Relationships/bpe_outputs/formal.nmt_baseline.bpe'],
            'nmt_copy': ['./data/Family_Relationships/bpe_outputs/formal.nmt_copy.bpe'],
            'nmt_combined': ['./data/Family_Relationships/bpe_outputs/formal.nmt_combined.bpe'],
        }
    embedding_path = './new_exp_em/embedding/embedding.bpe.big.txt'
    embedding, vocab_hash = embedding_api.load_word_embedding(embedding_path)
    nn = NNModel(np.array(embedding),mode='eval')
    nn.batch_size = 128
    nn.build_basic_rnn_model()
    for key in files.keys():
        data=preprocess(informal_src_list=[],formal_src_list=files[key],embedding_path=embedding_path,shuffle=False)
        result = nn.predict_prob([t.x for t in data], model_path='./new_exp_em/classifier/model/600model.ckpt')
        base_name=os.path.basename(files[key])
        with open(os.path.join(output_dir,base_name+'.formality_score'),'w',encoding='utf-8') as fw:
            for r in result:
                fw.write(str(r[1])+'\n')


def evaluate_formality(resources):
    eval_log={}
    for key in resources.keys():
        eval_log[key] = test_formality_score(resources[key])
    return eval_log