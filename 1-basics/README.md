# 환경별로 state가 무엇인지, action은 무엇인지를 정리

# Classic Control : Cartpole-v?

haha

# MuJoCo : Walker2d-v?

* state
	* 위치, 속도 정보가 들어가는 듯?
	* https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v3.py
```
def _get_obs(self):
	position = self.sim.data.qpos.flat.copy()
	
	velocity = np.clip(
		self.sim.data.qvel.flat.copy(), -10, 10)

	if self._exclude_current_positions_from_observation:
		position = position[1:]

	observation = np.concatenate((position, velocity)).ravel()
	
	return observation
```
* action : 
	* 왠지... 아래의 6가지 정보가 될 것 같은데...?
	* https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/walker2d.xml
```
<actuator>
<!-- <motor joint="torso_joint" ctrlrange="-100.0 100.0" isctrllimited="true"/>-->
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_joint"/>
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_joint"/>
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_joint"/>
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_left_joint"/>
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_left_joint"/>
<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_left_joint"/>
<!-- <motor joint="finger2_rot" ctrlrange="-20.0 20.0" isctrllimited="true"/>-->
</actuator>
```	
* reward
	* `forward_reward + healthy_reward - costs` 로 계산하네
	* https://github.com/openai/gym/blob/master/gym/envs/mujoco/walker2d_v3.py
```
def step(self, action):
	x_position_before = self.sim.data.qpos[0]
	self.do_simulation(action, self.frame_skip)
	x_position_after = self.sim.data.qpos[0]
	x_velocity = ((x_position_after - x_position_before) / self.dt)

	ctrl_cost = self.control_cost(action)

	forward_reward = self._forward_reward_weight * x_velocity
	healthy_reward = self.healthy_reward

	rewards = forward_reward + healthy_reward
	costs = ctrl_cost

	observation = self._get_obs()
	reward = rewards - costs
	done = self.done
	info = {
		'x_position': x_position_after,
		'x_velocity': x_velocity,
	}

	return observation, reward, done, info
```	