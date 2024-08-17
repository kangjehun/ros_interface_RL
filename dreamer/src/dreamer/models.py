import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
import tensorflow.keras.mixed_precision as prec

import tools as tools


class RSSM(tools.Module):

    def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
        super().__init__()
        self._name = "rssm"
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._cell = tfkl.GRUCell(self._deter_size)
    
    def initial(self, batch_size):
        dtype = prec.global_policy().compute_dtype
        return dict(
            mean=tf.zeros([batch_size, self._stoch_size], dtype),
            std=tf.zeros([batch_size, self._stoch_size], dtype),
            stoch=tf.zeros([batch_size, self._stoch_size], dtype),
            deter=self._cell.get_initial_state(None, batch_size, dtype))
    
    @tf.function
    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0]) # initialize with batch size
        embed = tf.transpose(embed, [1, 0, 2])
        action = tf.transpose(action, [1, 0, 2])
        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs),
            (action, embed), (state, state)
        )
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()}
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return post, prior
    
    @tf.function
    def obs_step(self, prev_state, prev_action, embed):
        prior = self.img_step(prev_state, prev_action)
        x = tf.concat([prior['deter'], embed], -1)
        x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
        return post, prior

    @tf.function
    def img_step(self, prev_state, prev_action):
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x, deter = self._cell(x, [prev_state['deter']])
        deter = deter[0]  # Keras wraps the state in a list.
        x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_dist({'mean': mean, 'std': std}).sample()
        prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
        return prior

    @staticmethod
    def get_dist(state):
        # return a multivariate normal distribution
        return tfd.MultivariateNormalDiag(state['mean'], state['std'])
    
    @staticmethod
    def get_feat(state):
        return tf.concat([state['stoch'], state['deter']], -1)
    
    @tf.function
    def imagine(self, action, state=None):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = tf.transpose(action, [1, 0, 2])
        prior = tools.static_scan(self.img_step, action, state)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior
        
class ActionDecoder(tools.Module):

    def __init__(self, size, layers, units, dist='tanh_normal', act=tf.nn.elu, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self._name = "actor"
        self._size = size
        self._layers = layers
        self._units = units
        self._dist = dist
        self._act = act
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

    def __call__(self, features, training=False):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = features
        for index in range(self._layers):
            x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
        if self._dist == 'tanh_normal':  # Original from Dreamer
            # https://www.desmos.com/calculator/rcmcf5jwe7
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            mean, std = tf.split(x, 2, -1)
            mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
            std = tf.nn.softplus(std + raw_init_std) + self._min_std
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        elif self._dist == 'normalized_tanhtransformed_normal':
            # Normalized variation of the original actor: (mu,std) normalized, then create tanh normal from them
            # The normalization params (moving avg, std) are updated only during training
            x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
            x = tf.reshape(x, [-1, 2 * self._size])
            x = self.get(f'hnorm', tfkl.BatchNormalization)(x, training=training)  # `training` true only in imagination
            x = tf.reshape(x, [*features.shape[:-1], -1])
            mean, std = tf.split(x, 2, -1)
            std = tf.nn.softplus(std) + self._min_std  # to have positive values
            dist = tfd.Normal(mean, std)
            dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
            dist = tfd.Independent(dist, 1)
            dist = tools.SampleDist(dist)
        else:
            raise NotImplementedError(self._dist)
        return dist
