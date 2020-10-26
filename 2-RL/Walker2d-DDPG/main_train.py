# DDPG main

import gym
from ddpg_agent import DDPGagent

def main():

    max_episode_num = 3000
    env = gym.make("Walker2d-v2")
    agent = DDPGagent(env, do_train=True)

    agent.train(max_episode_num+1)

    agent.plot_result()


if __name__=="__main__":
    main()