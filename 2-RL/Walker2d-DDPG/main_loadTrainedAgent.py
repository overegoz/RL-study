# DDPG main

import gym
from ddpg_agent import DDPGagent

def main():

    env = gym.make("Walker2d-v2")
    agent = DDPGagent(env, do_train=False)

    agent.actor.load_weights('./save_weights/')
    agent.critic.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    while True:
        env.render()
        action = agent.actor.predict(state)
        state, reward, done, _ = env.step(action)
        time += 1

        print('Time: ', time, 'Reward: ', reward)

        #if done: break
        if done: state = env.reset()

    env.close()

if __name__=="__main__":
    main()