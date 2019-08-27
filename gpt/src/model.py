import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
from utils.common import gather_2d,tile_to_beam_size,merge_first_two_dims
from tensorflow.python.util import nest

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))
        h2 = conv1d(h, 'c_proj', nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


class Encoder():
    def __init__(self,scope,hparam):
        if scope is None:
            self.scope='encoder'
        else:
            self.scope=scope
        self.hparams=hparam

    def encode(self, h, h_len, past=None, scope='encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                # Transformer
                presents = []
                pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
                assert len(pasts) == self.hparams.n_layer
                for layer, past_one in enumerate(pasts):
                    h, present = block(h, 'h%d' % layer, past=past_one, hparams=self.hparams)
                    presents.append(present)
                presents = tf.stack(presents, axis=1)
                h = norm(h, 'ln_f')
                final_id = h_len - 1
                h = gather_2d(h, tf.expand_dims(final_id, axis=1))
                target_mask = tf.sequence_mask(h_len-1, maxlen=tf.shape(h)[1], dtype=tf.float32)#h_len-1把sentence token给mask掉
                target_mask = tf.expand_dims(target_mask, 2)
                encode_out = tf.transpose(presents, perm=(0, 4, 2, 3, 1, 5))
                ori_enc_shape = tf.shape(encode_out)
                encode_out = tf.reshape(encode_out, shape=(tf.shape(presents)[0], tf.shape(presents)[4], -1))
                encode_out = tf.multiply(encode_out, target_mask)
                encode_out = tf.reshape(encode_out, shape=ori_enc_shape)
                encode_out = tf.transpose(encode_out, perm=(0, 4, 2, 3, 1, 5))
                encode_out.set_shape(past_shape(hparams=self.hparams, batch_size=None))
                return encode_out, h



class Decoder():
    def __init__(self,scope,hparams):
        self.scope = scope
        self.hparams = hparams
        with tf.variable_scope(scope):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                self.wpe=tf.get_variable('wpe', [self.hparams.n_ctx, self.hparams.n_embd],
                                   initializer=tf.random_normal_initializer(stddev=0.01))
                self.wte = tf.get_variable('wte', [self.hparams.n_vocab, self.hparams.n_embd],
                                   initializer=tf.random_normal_initializer(stddev=0.02))
                self.attn_w = tf.get_variable(shape=(self.hparams.n_embd, self.hparams.n_embd), name='sen_attn_w')


    #def decode_all
    def decode_all(self,tokens,past_list,enc_h_list):
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
                results = {}
                if type(past_list)!=list:
                    past_list=[past_list]
                batch, sequence = shape_list(tokens)
                #past_length = 0
                all_past_length=[0 if past_list[0] is None else tf.shape(past_list[0])[-2]]
                past_length = tf.reduce_max(tf.stack(all_past_length,axis=0),axis=0)
                h = tf.gather(self.wte, tokens) + tf.gather(self.wpe, positions_for(tokens, past_length))
                values_present = {}
                for i in range(0, self.hparams.n_layer):
                    querys = h
                    values_h = []
                    for j in range(0, len(past_list)):
                        past = past_list[j]
                        pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
                        assert len(pasts) == self.hparams.n_layer
                        h, present = block(querys, 'h%d' % i, past=pasts[i], hparams=self.hparams)
                        values_h.append(h)
                        if j in values_present:
                            values_present[j].append(present)
                        else:
                            values_present[j]=[present]
                    enc_h_all = tf.concat(enc_h_list, axis=1)
                    attn_score = tf.tensordot(querys, self.attn_w, axes=(2, 0))
                    attn_score = tf.matmul(attn_score, tf.transpose(enc_h_all, perm=(0, 2, 1)))  # batch*seq*context_num
                    attn_score = tf.nn.softmax(attn_score,axis=2)
                    val_h_cat = tf.stack(values_h, axis=2)
                    val_h_cat = tf.expand_dims(attn_score, axis=3) * val_h_cat
                    val_h_cat = tf.reduce_sum(val_h_cat, axis=2)
                    h = val_h_cat
                for j in range(0,len(past_list)):
                    values_present[j]=tf.stack(values_present[j],axis=1)
                    past_list[j]=tf.concat([past_list[j],values_present[j]],axis=-2)
                h = norm(h, 'ln_f')
                # Language model loss.  Do tokens <n predict token n?
                h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embd])
                logits = tf.matmul(h_flat, self.wte, transpose_b=True)
                logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
                results['logits'] = logits
                return results


    def sef_var_for_beam_search(self,enc_0_len,enc_h_list,beam_size):
        self.enc_0_len=enc_0_len
        self.enc_h_list=enc_h_list
        self.enc_h_all = tf.concat(self.enc_h_list, axis=1)
        self.enc_h_all=merge_first_two_dims(tile_to_beam_size(self.enc_h_all,beam_size=beam_size))


    def decode_one_step(self,hparams:"no use, only for consistency of api", input_token, past_dec:list):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
                all_past_length = [0 if past_dec[j] is None else tf.shape(past_dec[j])[-2] for j in range(0,len(past_dec))]
                past_length=tf.reduce_max(tf.stack(all_past_length,axis=0),axis=0)
                h = tf.gather(self.wte, input_token) + tf.gather(self.wpe, positions_for(input_token, past_length))
                results = {}
                batch, sequence = shape_list(input_token)
                values_present = {}
                for i in range(0, self.hparams.n_layer):
                    querys = h
                    values_h = []
                    for j in range(0, len(past_dec)):
                        dec_pasts = tf.unstack(past_dec[j], axis=1) if past_dec[j] is not None else [None] * self.hparams.n_layer  #
                        h, present = block(querys, 'h%d' % i,
                                           past=dec_pasts[i],
                                           hparams=self.hparams)
                        values_h.append(h)
                        if j in values_present:
                            values_present[j].append(present)
                        else:
                            values_present[j]=[present]
                    attn_score = tf.tensordot(querys, self.attn_w, axes=(2, 0))
                    attn_score = tf.matmul(attn_score, tf.transpose(self.enc_h_all, perm=(0, 2, 1)))  # batch*seq*context_num
                    attn_score = tf.nn.softmax(attn_score, axis=2)
                    val_h_cat = tf.stack(values_h, axis=2)
                    val_h_cat = tf.expand_dims(attn_score, axis=3) * val_h_cat
                    val_h_cat = tf.reduce_sum(val_h_cat, axis=2)
                    h = val_h_cat
                for j in range(0,len(past_dec)):
                    values_present[j]=tf.stack(values_present[j],axis=1)
                    past_dec[j]=tf.concat([past_dec[j],values_present[j]],axis=-2)
                h = norm(h, 'ln_f')
                # Language model loss.  Do tokens <n predict token n?
                h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embd])
                logits = tf.matmul(h_flat, self.wte, transpose_b=True)
                logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
                results['logits'] = logits
                results['presents']= past_dec
                return results


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)
        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X, name='gggggg1') + tf.gather(wpe, positions_for(X, past_length),name='ggggggg2')
        #h=tf.gather(wpe, positions_for(X, past_length),name='ggggggg2')
        #h=tf.add(tf.nn.embedding_lookup(wte, X, name='gggggg1'),tf.nn.embedding_lookup(wpe, positions_for(X, past_length), name='ggggggg2'))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f')

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
