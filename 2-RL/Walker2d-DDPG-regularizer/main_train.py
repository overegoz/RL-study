# DDPG main

import gym
from ddpg_agent import DDPGagent

def main():

    max_episode_num = 200
    env = gym.make("Walker2d-v2")
    agent = DDPGagent(env)

    agent.train(max_episode_num)

    agent.plot_result()



if __name__=="__main__":
    main()