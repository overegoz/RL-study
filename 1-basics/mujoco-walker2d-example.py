import gym
import time

env = gym.make("Walker2d-v2")
# env = env.unwrapped
# end.seed(1)

RENDER_ENV = True
NUM_EPISODES = 10

"""
Something to know about the Walker2d-v2 environment:
- state
  . state.shape = (17,0)
  . state.shape[0] = 17 // 17 items in each state
- action
  . action.shape = (6,0)
  . action.shape[0] = 6 // 6 items in each action
"""

if __name__ == "__main__":
    for i in range(NUM_EPISODES):
        state = env.reset()  # initialize the env, and get the observation
        done = False
        while not done: 
            if RENDER_ENV: env.render()

            # choo a random action
            action = env.action_space.sample()
            print('action: ', action)

            # apply the action, and then get (state, reward, done-flag)
            state, reward, done, _ = env.step(action)
            print('state: ', state)

