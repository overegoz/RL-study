# 설치하기

# open ai gym 설치하기

* 설치방법 : `pip install gym`, https://gym.openai.com/docs/
* 설치확인 : 1-basics/classic-cartpole-example.py 실행

# mujoco-py 설치하기

* mujoco는 공짜가 아니다. 단, 학교 이메일로 인증 받으면 매년 갱신하는 무료 라이선스를 받을 수 있다.
* 설치방법 : https://github.com/openai/mujoco-py
* 설치방법 : https://seunghyun-lee.tistory.com/2
  * 근데, https://www.roboti.us/index.html 가니까 mujoco200 win64가 있는데?
* 설치파일 : https://www.roboti.us/index.html
* 설치확인 : 
```
$ python3
>> import mujoco_py
>> import os
>> mj_path, _ = mujoco_py.utils.discover_mujoco()
>> xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
>> model = mujoco_py.load_model_from_path(xml_path)
>> sim = mujoco_py.MjSim(model)

>> print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

>> sim.step()
>> print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]
```
