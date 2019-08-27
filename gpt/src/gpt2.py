#!/usr/bin/env python3
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
from gpt.src import model, encoder, sample
from gpt.src import beamsearch


class GPT2:
    def __init__(self,config_path='./models/117M'):
        self.config_path = config_path
        self.text_enc = encoder.get_encoder(self.config_path)
        self.hparams = model.default_hparams()
        with open(os.path.join(self.config_path, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))
        self.eos_id = self.text_enc.encode('\n')[0]



    def ensemble_decoding_beam_search_graph(self,context_list,beam_size,batch_size,max_decode_length,eos_id,model_num,decode_alpha=0.6):
        def step(hparams, tokens, past=None, scope=None):
            if scope is not None:
                with tf.variable_scope(scope):
                    lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
                    logits = lm_output['logits']
                    presents = lm_output['present']
                    presents.set_shape(model.past_shape(hparams=hparams, batch_size=None))
                    return {
                        'logits': logits,
                        'presents': presents,
                    }
        context_output_list=[]
        context_state_list=[]
        all_scopes=[]
        for i in range(0,model_num):
            with tf.variable_scope('model_'+str(i)) as sc:
                with tf.name_scope('sample_sequence'):
                    context_output_list.append(step(self.hparams, context_list[i][:, :-1],scope=sc))
                    context_state_list.append(context_output_list[-1]['presents'])
            all_scopes.append('model_'+str(i))
        with tf.name_scope('beam_search'):
            init_seq = tf.expand_dims(context_list[0][:, -1], axis=1)
            seqs, scores = beamsearch.create_inference_graph(init_seqs=init_seq, state=context_state_list,
                                                                 step_fn=step, hparams=self.hparams,
                                                                 decode_length=max_decode_length,
                                                                 batch_size=batch_size, beam_size=beam_size,
                                                                 decode_alpha=decode_alpha, eos_id=eos_id, scopes_for_ensemble=all_scopes,
                                                             ensemble=True, concat_state_dim=None)
        return seqs, scores



    def build_beam_search_graph(self,beam_size,batch_size,max_decode_length,decode_alpha=0.6):
        self.inputs = tf.placeholder(tf.int32, [1, None])
        def step(hparams, tokens, past=None):
            lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
            logits = lm_output['logits']
            presents = lm_output['present']
            presents.set_shape(model.past_shape(hparams=hparams, batch_size=None))
            return {
                'logits': logits,
                'presents': presents,
            }
        with tf.name_scope('sample_sequence'):
            context_output = step(self.hparams, self.inputs[:, :-1])
            context_state=context_output['presents']
        with tf.name_scope('beam_search'):
            init_seq = tf.expand_dims(self.inputs[:, -1], axis=1)
            seqs, scores=beamsearch.create_inference_graph(init_seqs=init_seq,state=context_state,
                                              step_fn=step,hparams=self.hparams,
                                              decode_length=max_decode_length,
                                              batch_size=batch_size,beam_size=beam_size,
                                              decode_alpha=decode_alpha,eos_id=self.eos_id,
                                              ensemble=False, concat_state_dim=-2)

        return seqs,scores

    def build_inferring_graph(self, context, seed=None, nsamples=1,
                                             batch_size=1, length=None, temperature=1, top_k=40):
        self.generate_batch_size = batch_size
        self.generate_n_samples = nsamples
        if batch_size is None:
            batch_size = 1

        if length is None:
            length = self.hparams.n_ctx // 2
        elif length > self.hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % self.hparams.n_ctx)

        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=self.hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )
        return output


    def build_training_graph(self,input,input_len,target,target_mask=None):
        batch_max_seq_len = tf.shape(input)[1]
        def step(hparams, tokens, past=None):
            lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
            logits = lm_output['logits']
            presents = lm_output['present']
            presents.set_shape(model.past_shape(hparams=hparams, batch_size=None))
            return {
                'logits': logits,
                'presents': presents,
            }
        with tf.name_scope('sample_sequence'):
            all_logits = step(hparams=self.hparams, tokens=input)['logits']
        with tf.name_scope('loss'):
            if target_mask is None:
                target_mask = tf.sequence_mask(input_len, maxlen=batch_max_seq_len, dtype=tf.float32)
        cost = sequence_loss(logits=all_logits, targets=target,
                             weights=target_mask)
        return cost










