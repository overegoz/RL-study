# DDPG Critic

from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.optimizers import Adam
from keras import regularizers
from tensorflow.keras import initializers
from keras.layers import BatchNormalization

import tensorflow as tf


class Critic(object):
    """
        Critic Network for DDPG: Q function approximator
    """
    def __init__(self, sess, state_dim, action_dim, tau, learning_rate):
        self.sess = sess

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.learning_rate = learning_rate

        # create critic and target critic network
        self.model, self.states, self.actions = self.build_network()
        self.target_model, _, _ = self.build_network()

        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')
        self.target_model.compile(optimizer=Adam(self.learning_rate), loss='mse')

        # compute dq_da to feed to the actor
        self.q_grads = tf.gradients(self.model.output, self.actions)

    ## critic network
    def build_network(self):
        state_input = Input((self.state_dim,))
        state_input_bn = BatchNormalization()(state_input)
        x1 = Dense(256, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(state_input_bn)   
        x2 = Dense(128, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(x1)
        x3 = Dense(64, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(x2)

        action_input = Input((self.action_dim,))
        # action은 이미 [-1, 1] 사이의 값을 가지고 있으니까, batch normalization을 따로 하지 않아도 될듯
        # 참고:  배치 정규화는 인공신경망에 입력값을 평균 0, 분산 1로 정규화(normalize)
        # from https://buomsoo-kim.github.io/keras/2018/04/24/Easy-deep-learning-with-Keras-5.md/
        #action_input_bn = BatchNormalization()(action_input)
        a1 = Dense(128, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(action_input)
        a2 = Dense(64, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(a1)

        h1 = concatenate([x3, a2], axis=-1)  # h2 = Add()([x2, a1])
        h2 = BatchNormalization()(h1)
        h3 = Dense(64, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(h2)
        h4 = Dense(16, activation='relu', \
                    kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                    bias_initializer=initializers.Zeros())(h3)
        h4_bn = BatchNormalization()(h4)
        # final output layer
        q_output = Dense(1, activation='linear', \
                        kernel_initializer=initializers.RandomNormal(stddev=0.5), \
                        bias_initializer=initializers.Zeros())(h4_bn)

        model = Model([state_input, action_input], q_output)
        model.summary()
        return model, state_input, action_input


    ## q-value prediction of target critic
    def target_predict(self, inp):
        return self.target_model.predict(inp)


    ## transfer critic weights to target critic with a aau
    def update_target_network(self):
        phi = self.model.get_weights()
        target_phi = self.target_model.get_weights()
        for i in range(len(phi)):
            target_phi[i] = self.tau * phi[i] + (1 - self.tau) * target_phi[i]
        self.target_model.set_weights(target_phi)


    ## gradient of q-values wrt actions
    def dq_da(self, states, actions):
        return self.sess.run(self.q_grads, feed_dict={
            self.states: states,
            self.actions: actions
        })

    ## single gradient update on a single batch data
    def train_on_batch(self, states, actions, td_targets):
        return self.model.train_on_batch([states, actions], td_targets)

    ## save critic weights
    def save_weights(self, path):
        self.model.save_weights(path)


    ## load critic wieghts
    def load_weights(self, path):
        self.model.load_weights(path + 'critic.h5')