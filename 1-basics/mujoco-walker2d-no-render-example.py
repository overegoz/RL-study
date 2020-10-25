import gym
import time

"""
Something to know about the Walker2d-v2 environment:
- state
  . state.shape = (17,0)
  . state.shape[0] = 17 // 17 items in each state
- action
  . action.shape = (6,0)
  . action.shape[0] = 6 // 6 items in each action
"""

"""
Results:

state space:
17
17
state_bound (high):  [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf]
state_bound (low):  [-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf
 -inf -inf -inf]
random action:  [ 0.59163535 -0.5008823   0.39323384  0.80062795  0.5794247  -0.6511662 ]
action space:
6
6
action_bound (high):  [1. 1. 1. 1. 1. 1.]
action_bound (low):  [-1. -1. -1. -1. -1. -1.]
"""

if __name__ == "__main__":
    env = gym.make("Walker2d-v2")
    # env = env.unwrapped
    # end.seed(1)

    state = env.reset()  # initialize the env, and get the observation
    
    # state space
    print('state space: ')
    print(env.observation_space.shape[0])
    print(state.shape[0])
    
    # get state bound
    print('state_bound (high): ', env.observation_space.high)
    print('state_bound (low): ', env.observation_space.low) 
    
    # choo a random action
    action = env.action_space.sample()
    print('random action: ', action)
    
    # action space
    print('action space: ')
    print(action.shape[0])
    print(env.action_space.shape[0])
    
    # get action bound
    print('action_bound (high): ', env.action_space.high)
    print('action_bound (low): ', env.action_space.low)    

    # apply the action, and then get (state, reward, done-flag)
    state, reward, done, _ = env.step(action)

