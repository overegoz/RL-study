# 설치하기

# open ai gym 설치하기

* 설치방법 : `pip install gym`, https://gym.openai.com/docs/
* 설치확인 : 1-basics/classic-cartpole-example.py 실행

# mujoco-py 설치하기

* mujoco는 공짜가 아니다. 단, 학교 이메일로 인증 받으면 매년 갱신하는 무료 라이선스를 받을 수 있다.
* 설치방법 : https://github.com/openai/mujoco-py
* 설치방법 : https://seunghyun-lee.tistory.com/2
  * 근데, https://www.roboti.us/index.html 가니까 mujoco200 win64가 있는데?
* 설치방법 (Win10, MuJoCo-200)
  * https://www.roboti.us/license.html 에서 학생용 라이선스 신청 (학교 이메일 계정 사용)
  * 24시간 내에 `MuJoCo Pro Personal Account` 라는 이메일을 받는데, 그 안에 account number가 text로 적혀있음
  * https://www.roboti.us/license.html 페이지에서 https://www.roboti.us/getid/getid_win64.exe 파일을 다운 받아서 실행하면 computer id를 알려줌
  * https://www.roboti.us/license.html 페이지에서 account number랑 computer id를 입력하면 activation key에 해당하는 txt 파일을 이메일로 받음
  * mujoco 파일을 다운로드 : https://www.roboti.us/index.html (설치하는 건 없고, 압출만 풀면 됨)
  * 압축해제 후, c:/Users/[본인계정]/.mujoco/mujoco200 경로에 파일을 저장 (리눅스라면 `~/.mujoco/mujoco200`)
  * act-key 파일 'mjkey.txt'을 .mujoco, .mujoco/mujoco200, .mujoco/mujoco200/bin 에 저장
  * 설치 확인 : C:\Users\[본인계정]\.mujoco\mjoco200\bin> 경로에서 simulate ../model/humanoid.xml를 실행해서, humanoid가 나오는지 확인
  * 환경변수 설정: 
    * bin 디렉토리를 PATH 환경변수에 저장
    * MUJOCO_HOME=C:\Users\hallym\.mujoco
    * MUJOCO_LICENSE_PATH=C:\Users\hallym\.mujoco
    * MUJOCO_PY_MJKEY_PATH=C:\Users\hallym\.mujoco\mjkey.txt
    * MUJOCO_PY_MUJOCO_PATH=C:\Users\hallym\.mujoco\mujoco200
  * MuJoCo-py 설치 : `pip3 install -U 'mujoco-py<2.1,>=2.0'`
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
