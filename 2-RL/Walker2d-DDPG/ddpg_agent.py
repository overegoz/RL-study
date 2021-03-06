# DDPG Agent for training and evaluation

import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
import random

from ddpg_actor import Actor
from ddpg_critic import Critic
from replaybuffer import ReplayBuffer

class DDPGagent(object):

    def __init__(self, env):

        self.sess = tf.Session()
        K.set_session(self.sess)

        self.SAVE_FREQ = 100 # save for 100 epi
        # 타겟 신경망을 일정 확률로 업데이트 해서, 학습 안정성을 높임
        self.target_network_update_rate = 0.5
        _ = input('Target network update rate is set to 0.3 : [ENTER]')
        # 노이즈를 추가하면 exploration이 되는데, 지금 학습하려는 control문제는
        # 그 자체로 워낙 복잡해서 노이즈가 없어도 될듯...?
        self.noise_add_rate = 0.0
        _ = input('Noise add rate is set to 0 : [ENTER]')

        ## hyperparameters
        #self.GAMMA = 0.95  # 보상에 적용될 discount factor (감가율)
        self.GAMMA = 0.99  # 보상에 적용될 discount factor (감가율)
        self.BATCH_SIZE = 128  # 한번 학습할때 몇개의 샘플을 사용할지
        #self.BUFFER_SIZE = 30000  # 샘플을 저장할 버퍼의 크기. 샘플을 버퍼에 넣고, 그 중에서 무작위로 골라서 학습함
        self.BUFFER_SIZE = 10000  # 샘플을 저장할 버퍼의 크기. 샘플을 버퍼에 넣고, 그 중에서 무작위로 골라서 학습함
        self.MIN_SAMPLES_TO_BEGIN_LEARNING = self.BATCH_SIZE * 10  # 학습 시작을 위해 필요한 최소 샘플 수
        self.ACTOR_LEARNING_RATE = 0.0001  # 액터(정책)의 학습률
        #self.ACTOR_LEARNING_RATE = 0.001  # 액터(정책)의 학습률
        self.CRITIC_LEARNING_RATE = 0.001  # 크리틱(Q함수)의 학습률
        #self.TAU = 0.01  # 신경망 업데이트 정도 (매 timestep마다 target 신경망을 조금씩 업데이트)
        self.TAU = 0.001  # 신경망 업데이트 정도 (매 timestep마다 target 신경망을 조금씩 업데이트)

        self.env = env
        # get state dimension
        self.state_dim = env.observation_space.shape[0]
        print('state_dim: ', self.state_dim)
        # get action dimension
        self.action_dim = env.action_space.shape[0]
        print('action_dim: ', self.action_dim)
        # get action bound
        self.action_bound = env.action_space.high
        print('action_bound: ', self.action_bound)

        ## create actor and critic networks
        self.actor = Actor(self.sess, self.state_dim, \
                           self.action_dim, self.action_bound[0], self.TAU, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.sess, self.state_dim, self.action_dim, self.TAU, self.CRITIC_LEARNING_RATE)

        ## initialize for later gradient calculation
        self.sess.run(tf.global_variables_initializer())  #<-- no problem without it

        ## initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []

    ## Ornstein Uhlenbeck Noise
    #def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):  # 오류가 너무 큰거 아닌가..
    def ou_noise(self, x, rho=0.015, mu=0, dt=1e-1, sigma=0.02, dim=1):
        return x + rho*(mu - x)*dt + sigma*np.sqrt(dt)*np.random.normal(size=dim)

    ## computing TD target: y_k = r_k + gamma*Q(s_k+1, a_k+1)
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]: y_k[i] = rewards[i]
            else: y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## train the agent
    def train(self, max_episode_num):

        # initial transfer model weights to target model network
        self.actor.update_target_network()
        self.critic.update_target_network()

        for ep in range(int(max_episode_num)):
            # reset OU noise
            pre_noise = np.zeros(self.action_dim)
            # reset episode
            time, episode_reward, done = 0, 0, False
            # reset the environment and observe the first state
            state = self.env.reset()

            while not done:
                # visualize the environment
                #self.env.render()
                # pick an action: shape = (1,)
                action = self.actor.predict(state)
                #print('action: ', action)
                noise = self.ou_noise(pre_noise, dim=self.action_dim)  # 노이즈가 너무 커서, 약간 줄였음...

                if random.random() < self.noise_add_rate:  # 일정 확률로 노이즈를 추가
                    # clip continuous action to be within action_bound
                    #print('before: ', action)
                    action = np.clip(action + noise, -self.action_bound, self.action_bound)
                    #print('after: ', action)

                # observe reward, new_state
                next_state, reward, done, _ = self.env.step(action)
                #print('reward: ', reward)
                # add transition to replay buffer
                #train_reward = (reward + 8) / 8  # 책에서 사용하는 공식인데, walker2d에선 필요 없음
                train_reward = reward
                self.buffer.add_buffer(state, action, train_reward, next_state, done)

                # start train after buffer has some amounts
                if self.buffer.get_buffer_size() > self.MIN_SAMPLES_TO_BEGIN_LEARNING:  
                    #print('train...in prog')

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, dones = self.buffer.sample_batch(self.BATCH_SIZE)
                    # predict target Q-values
                    target_qs = self.critic.target_predict([next_states, self.actor.target_predict(next_states)])
                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs, dones)
                    # train critic using sampled batch
                    self.critic.train_on_batch(states, actions, y_i)
                    # Q gradient wrt current policy
                    s_actions = self.actor.model.predict(states) # shape=(batch, 1),
                    # caution: NOT self.actor.predict !
                    # self.actor.model.predict(state) -> shape=(1,1)
                    # self.actor.predict(state) -> shape=(1,) -> type of gym action
                    s_grads = self.critic.dq_da(states, s_actions)
                    dq_das = np.array(s_grads).reshape((-1, self.action_dim))
                    # train actor
                    self.actor.train(states, dq_das)
                    # update both target network
                    # 일정 확률로 target network를 업데이트
                    # 학습 안정성을 높이기 위해...
                    if random.random() < self.target_network_update_rate:  
                        self.actor.update_target_network()
                        self.critic.update_target_network()


                # update current state
                pre_noise = noise
                state = next_state
                episode_reward += reward
                time += 1

            ## display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)

            self.save_epi_reward.append(episode_reward)

            ## save weights for 100 epi
            if (ep > 0) and (ep % 100 == 0):
                #print('Now save')
                self.actor.save_weights("./save_weights/actor-" + str(ep) + ".h5")
                self.critic.save_weights("./save_weights/critic-" + str(ep) + ".h5")

        np.savetxt('./save_weights/epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

