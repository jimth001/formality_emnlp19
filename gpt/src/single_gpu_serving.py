import tensorflow as tf
from gpt.src.gpt2 import GPT2
import numpy as np
import os
import tensorflow.contrib.slim as slim

class ensemble_beam_search_generator():
    def __init__(self,model_fn:GPT2,beam_size,model_directorys,max_dec_len=40,dec_alpha=0.6):
        self.model_fn=model_fn
        self.graph=tf.Graph()
        self.beam_size=beam_size
        self.max_decode_len=max_dec_len
        self.decode_alpha=dec_alpha
        self.model_path=model_directorys

    def build_graph_and_restore(self,eos_id,model_num):
        assert model_num==len(self.model_path)
        with self.graph.as_default():
            self.context = [tf.placeholder(tf.int32, [1, None]) for i in range(0,model_num)]
            self.seqs, _ =self.model_fn.ensemble_decoding_beam_search_graph(self.context,self.beam_size,1,self.max_decode_len,eos_id=eos_id,model_num=model_num,decode_alpha=self.decode_alpha)
            all_var_list = slim.get_variables_to_restore()
            infer_vars_for_models=[[] for i in range(0,model_num)]
            for v in all_var_list:
                if 'Adam' in v.name:
                    pass
                elif v.name.startswith('beta'):
                    pass
                elif v.name.startswith('parallel'):
                    pass
                elif v.name.startswith('model_'):
                    strs=v.name.split('/')
                    id=int(strs[0].split('_')[-1])
                    infer_vars_for_models[id].append(v)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=self.graph, config=config)
            restore_ops=[]
            for ckpt,vars in zip(self.model_path,infer_vars_for_models):
                # Load checkpoints
                tf.logging.info("Loading %s" % ckpt)
                var_list = tf.train.list_variables(ckpt)
                values = {}
                reader = tf.train.load_checkpoint(ckpt)
                for (name, shape) in var_list:
                    if not name.startswith('model/'):  # ignore global_step
                        continue
                    tensor = reader.get_tensor(name)
                    values[name] = tensor
                for v in vars:
                    tmp='/'.join(v.name.split('/')[1:])
                    v_name=tmp.split(':')[0]
                    op=tf.assign(v,values[v_name])
                    restore_ops.append(op)
            sess.run(restore_ops)
            return sess

    def print_all_trainable_vars(self):
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

    def generate(self,sess,raw_texts,append_flag='\t'):
        assert len(raw_texts)==len(self.context)
        contexts_tokens = [self.model_fn.text_enc.encode(raw_text.strip()+append_flag) for raw_text in raw_texts]
        feed_dict={}
        for c,p in zip(contexts_tokens,self.context):
            feed_dict[p]=[c]
        seqs = sess.run(self.seqs, feed_dict=feed_dict)
        seqs=seqs[:,0,:]
        text=self.model_fn.text_enc.decode(seqs[0])
        return text


class teacher_force_generator():#no use,uncompleted
    def __init__(self,model_fn,beam_size,model_directory,max_dec_len=40,dec_alpha=0.6):
        self.model_fn=model_fn
        self.graph=tf.Graph()
        self.beam_size=beam_size
        self.max_decode_len=max_dec_len
        self.decode_alpha=dec_alpha
        self.model_path=model_directory

    def build_graph_and_restore(self,eos_id):
        with self.graph.as_default():
            self.context = tf.placeholder(tf.int32, [1, None])
            self.seqs, _ =self.model_fn.build_beam_search_graph(self.beam_size,1,self.max_decode_len,decode_alpha=self.decode_alpha)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.model_path)
            sess = tf.Session(graph=self.graph, config=config)
            saver.restore(sess, ckpt)
            return sess

    def print_all_trainable_vars(self):
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


    def generate(self,sess,raw_text,append_flag='\t',multi_pls=False):
        if multi_pls:
            strs=raw_text.split(append_flag)
            feed_dict={}
            for i in range(0,len(strs)):
                tokens=self.model_fn.text_enc.encode(strs[i])+self.model_fn.text_enc.encode('\a')
                l=len(tokens)
                feed_dict[self.model_fn.inputs[i]]=[tokens]
                feed_dict[self.model_fn.input_lens[i]]=[l]
            seqs = sess.run(self.seqs, feed_dict=feed_dict)
            seqs = seqs[:, 0, :]
            text = self.model_fn.text_enc.decode(seqs[0])
        else:
            context_tokens = self.model_fn.text_enc.encode(raw_text.strip() + append_flag)
            seqs = sess.run(self.seqs, feed_dict={
                self.context: [context_tokens]
            })
            seqs = seqs[:, 0, :]
            text = self.model_fn.text_enc.decode(seqs[0])
        return text


class beam_search_generator():
    def __init__(self,model_fn,beam_size,model_directory,max_dec_len=40,dec_alpha=0.6):
        self.model_fn=model_fn
        self.graph=tf.Graph()
        self.beam_size=beam_size
        self.max_decode_len=max_dec_len
        self.decode_alpha=dec_alpha
        self.model_path=model_directory

    def build_graph_and_restore(self,eos_id):
        with self.graph.as_default():
            #self.context = tf.placeholder(tf.int32, [1, None])
            self.seqs, _ =self.model_fn.build_beam_search_graph(self.beam_size,1,self.max_decode_len,decode_alpha=self.decode_alpha)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.model_path)
            sess = tf.Session(graph=self.graph, config=config)
            saver.restore(sess, ckpt)
            return sess

    def print_all_trainable_vars(self):
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


    def generate(self,sess,raw_text,append_flag='\t',multi_pls=False):
        if multi_pls:
            strs=raw_text.split(append_flag)
            assert len(strs)==len(self.model_fn.inputs)
            feed_dict={}
            for i in range(0,len(strs)):
                tokens=self.model_fn.text_enc.encode(strs[i])+self.model_fn.text_enc.encode('\t')
                l=len(tokens)
                feed_dict[self.model_fn.inputs[i]]=[tokens]
                feed_dict[self.model_fn.input_lens[i]]=[l]
            seqs = sess.run(self.seqs, feed_dict=feed_dict)
            seqs = seqs[:, 0, :]
            text = self.model_fn.text_enc.decode(seqs[0])
        else:
            context_tokens = self.model_fn.text_enc.encode(raw_text.strip() + append_flag)
            seqs = sess.run(self.seqs, feed_dict={
                self.model_fn.inputs: [context_tokens]
            })
            seqs = seqs[:, 0, :]
            text = self.model_fn.text_enc.decode(seqs[0])
        return text


class single_gpu_server():
    def __init__(self,model_fn:GPT2):
        self.generate_n_samples=1
        self.generate_batch_size=1
        self.model_fn=model_fn
        self.graph=tf.Graph()
        self.model_name='117M/model_infer'
        self.sequence_length = 130
        self.temperature = 1
        self.top_k=40
        self.seed=None

    def build_serving_graph_and_restore(self):
        with self.graph.as_default():
            self.context = tf.placeholder(tf.int32, [self.generate_batch_size, None])
            self.output=self.model_fn.build_inferring_graph(self.context,seed=self.seed,
                                                nsamples=self.generate_n_samples,batch_size=self.generate_batch_size,
                                                length=self.sequence_length,temperature=self.temperature,
                                                top_k=self.top_k)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join('models', self.model_name))
            sess = tf.Session(graph=self.graph, config=config)
            saver.restore(sess, ckpt)
            return sess


    def interactive_generate(self, sess, raw_text, print_log=False):
        assert self.generate_n_samples % self.generate_batch_size == 0
        context_tokens = self.model_fn.text_enc.encode(raw_text)
        generated = 0
        results = []
        if print_log:
            print("=" * 40 + " RawText " + str(generated) + " " + "=" * 40)
            print(raw_text)
        for _ in range(self.generate_n_samples // self.generate_batch_size):
            out = sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(self.generate_batch_size)]
            })[:, len(context_tokens):]
            for i in range(self.generate_batch_size):
                generated += 1
                text = self.model_fn.text_enc.decode(out[i])
                if print_log:
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
                results.append(text)
        if print_log:
            print("=" * 80)
        return results