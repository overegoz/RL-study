#
# Open AI Gym에서 cartpole 환경을 불러오고, 무작위로 행동하는 에이전트 구현
#
# https://gym.openai.com/envs/CartPole-v1/
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
import gym
import numpy as np
import random

env = gym.make('CartPole-v1')
goal_steps = 500

while True:
  obs = env.reset()
  for i in range(goal_steps):
    env.render()  # render the env

    action = env.action_space.sample() # 랜덤하게 행동을 선택
    obs, reward, done, info = env.step(action)  # 행동을 하고, 한 개의 time-step을 진행

    if done:  # 만약 episode가 끝 났다면...
      break
    