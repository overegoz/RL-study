# DDPG 기법을 사용해서 Mujoco-py Walker2d 에이전트 개발

notes
* Pendulum을 위한 DDPG 코드를 기준으로 작업중...
* 학습을 1000 epi 동안 해도, 성능이 크게 증가하지 않는다?

env의 action과 state...
* state
  * 17개의 값이 들어있는 벡터이다.
  * 임의의 상태를 출력해 봤는데, 값들이 서로 다른 범위(min, max)를 가지고 있는 것 같다.
* action
  * action은 6개의 값이 들어있는 벡터인데, 모두 [-1.0, 1.0] 범위 이내의 실수 값이다.
  * actor network에서 output layer의 activation을 tanh로 설정하였으니 문제 없을듯
  * actor network (입력 : 상태, 출력 : action)