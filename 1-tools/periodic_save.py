# http://keras-ko.kr/getting_started/faq/index.html

import os
from tensorflow import keras

# 체크포인트를 저장할 디렉토리를 준비합니다.
checkpoint_dir = './ckpt'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_model():
    # 새로운 선형 회귀 모델을 만듭니다.
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model


def make_or_restore_model():
    # 최신 모델을 복원하거나 체크포인트가 없으면 새로운 모델을 만듭니다.
    checkpoints = [checkpoint_dir + '/' + name
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('복원한 체크포인트:', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print('새로운 모델 생성')
    return make_model()


model = make_or_restore_model()
callbacks = [
    # 이 콜백은 100번의 배치마다 SavedModel 파일을 저장합니다.
    # 파일 이름에 훈련 손실이 포함되었습니다.
    keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir + '/ckpt-loss={loss:.2f}',
        save_freq=100)
]
model.fit(train_data, epochs=10, callbacks=callbacks)