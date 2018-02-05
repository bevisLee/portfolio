
## url - https://codeonweb.com/entry/d64e9da0-b783-477c-9f62-ad8903e8c4f3

"""
신경망과의 첫 만남

우리는 첫 번째 신경망 예제를 살펴보겠습니다. 이 예제는 Keras를 사용하여 숫자 손글씨 분류하기입니다.

Keras나 비슷한 다른 라이브러리를 사용해 본 경험이 없다면 첫 번째 예제를 바로 이해하기는 어려울 수도 있습니다. 걱정마세요, 우선 예제를 통해서 전체 큰 그림을 보고, 다음 장에서 이 예제의 요소들을 하나씩 살펴 보고 상세하게 설명할 것입니다.

그러니 어떤 단계가 제멋대로인 듯 보이고 심지어 마술 같아 보이더라도 걱정하지 마세요. 어쨌든 어디서든 시작은 해야하니까요.

keras가 본 실습환경에서 동작하는지 확인해 봅니다.
"""

import keras
print(keras.__version__)

"""
MNIST 숫자 손글씨 읽기

목적: 숫자 손글씨의 그레이스케일 이미지(28 픽셀 X 28 픽셀)를 0부터 9까지의 10개의 범주로 분류하는 것입니다.
데이터 셋: 사용할 데이터셋은 기계학습 학계 그 자체의 역사만큼이나 오래 되었고, 또 집중적으로 연구된 고전인 MNIST 데이터셋입니다. 이 데이터셋은 1980년대에 미국 국립표준기술연구소(NIST, 네 MNIST의 그 NIST입니다)에서 수집한 60,000개의 훈련 이미지와 10,000개의 시험 이미지로 구성되어 있습니다.
MNIST를 "푼다"의 의미는?
심층학습에 있어서는 "Hello World"와 같다고 보시면 됩니다. 알고리즘이 예상한 대로 작동하는지 확인하는 거죠. 기계학습 연구자라면 MNIST를 학술 논문이나 블로그 등에서 계속 보게 될 겁니다.
데이터 셋을 보자!

MNIST를 keras의 datasets을 이용해서 가져와 봅시다.

훈련 셋트 (training data sets)
모델이 학습할 "훈련 세트"는 손글시씨가 들어간 train_images와 손글씨가 어떤 숫자를 의미하는지 정답을 써 놓은 train_labels으로 구성됩니다. 이미지는 Numpy 배열로 인코딩되어 있고, 레이블은 0부터 9까지 숫자의 배열로 되어 있습니다. 그리고 각 이미지와 레이블은 일대일로 대응되어 있습니다.
"""


from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

"""
훈련 데이터를 한번 볼까요:
"""

print(train_images.shape)
print(len(train_labels))
print(train_labels)

"""
시험 셋트 (test data sets) 
학습된 모델은 test_images와 test_labels로 구성된 "시험 세트"에 대해서 시험을 거치게 됩니다.
시험 데이터도 한번 보죠:
"""

print(test_images.shape)
print(len(test_labels))
print(test_labels)

"""
신경망 학습

그럼 신경망을 쌓아볼까요!

여러분은 초보입니다. 아직은 이 예제의 모든 것을 이해하리라고 기대하지 않습니다. 그러니 안심하시길:D
"""

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

"""
신경망의 핵심 블록은 "계층"이란 데이터를 처리하는 모듈인데 데이터에 대한 "필터" 같은 거라고 생각하시면 됩니다. 어떤 데이터가 들어와서 좀 더 쓸만한 형태로 바뀌어서 나갑니다. 정확하게 말하자면, 계층에 주어지는 데이터에서 문제를 다루는 데에 있어 좀 더 의미가 있을 표현을 추출해 냅니다. 대부분의 심층학습은 실제로 점진적인 "데이터 증류" 형태를 구현하는 단순한 계층 간의 연결로 구성됩니다. 심층학습 모델은 데이터를 처리하는 체와 같이 점점 더 정교해지는 데이터 필터(바로 계층이죠)의 연속으로 이루어집니다.

우리 신경망은 두 개의 Dense 계층의 연속으로 이루어져 있습니다. Dense 계층은 '연결 밀도가 빽빽한(densely-connected)'('완전히 연결된, fully-connected'이라고도 합니다) 신경망 계층입니다. 두 번째(이자 마지막) 계층은 10개 범주의 "소프트맥스" 계층으로, 합계가 1이 되는 10개의 확률 점수의 배열을 반환합니다. 각 점수는 현재의 숫자 이미지가 10개 숫자 분류 각각에 속할 확률입니다.

우리 신경망을 훈련시킬 준비를 하기 위해 컴파일 단계로 3가지를 더 선택해야 합니다:

손실 함수: 훈련 데이터에 대해 얼마나 신경망이 얼마나 잘 맞아들어가고 있는지 측량하여 바람직한 방향으로 조정하는 척도입니다.
최적화기: 신경망이 받은 데이터와 손실 함수에 기반하여 스스로를 업데이트하는 메커니즘입니다.
메트릭: 훈련 및 시험 중에 모니터링할 측정값입니다. 여기서는 정확도(올바르게 분류된 이미지의 비율)만을 모니터링합니다.
손실 함수와 최적화기의 정확한 목적은 다음 두 장에 걸쳐 명쾌하게 알아보도록 하겠습니다.
"""

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

"""
훈련을 시작하기 전에, 데이터에 대해 신경망이 예상하는 형태로 바꾸고 모든 값을 [0, 1] 사이로 범위를 바꾸는 전처리를 해 줍니다. 예를 들어 훈련 데이터는 [0, 255] 범위 내의 uint8 타입 값이 (60000, 28, 28) 형태의 배열로 저장되어 있습니다. 이 데이터를 [0, 1] 범위 내의 float32 값의 (60000, 28 * 28) 형태 배열로 바꾸어 줍니다.
"""

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

"""
그리고 레이블을 범주 별로 인코딩 해 줍니다. 이 단계에 대해서는 3장에서 설명할 겁니다:
"""

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""
이제 우리 신경망을 훈련할 준비가 다 되었습니다. Keras에서는 신경망의 fit()메소드를 호출하여 훈련을 할 수 있습니다: 훈련 데이터에 대해서 모델을 "딱 맞추는(fit)" 것이죠.
"""

network.fit(train_images, train_labels, epochs=5, batch_size=128)

"""
훈련 중에는 두 개의 값이 표시됩니다: 훈련 데이터에 대한 신경망의 "손실"과 "정확도"입니다.
훈련 데이터에 대해서 빨리 0.989(즉 98.9%이죠!)의 정확도에 도달하였네요. 이제, 시험 데이터에도 우리 모델의 성능이 잘 나오는지 체크해 봅시다:
"""

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

"""
시험 세트에 대한 정확도는 97.8%로 나왔습니다. 훈련 세트에 대한 정확도 보다는 조금 낮군요. 훈련 정확도와 시험 정확도 사이의 이 차이는 "과적합"의 한 예입니다. 사실, 기계학습 모델은 훈련 데이터에 비해 새로운 데이터에 대해 성능이 떨어지는 경향이 있습니다. 과적합은 3장의 중심 주제가 될 겁니다.

이제 첫 번째 예제에 대해 결론을 내려볼까요. 우리는 이제 막 20줄도 안 되는 파이썬 코드로 손글씨 숫자를 분류하는 신경망을 어떻게 쌓고 훈련시키는지를 살펴보았습니다.

이 강의는 Keras를 만든이 Francois Chollet가 쓴 딥러닝학습서 "Deep Learning with Python (Manning Publications)"를 참조하여 만들었습니다. Deep Learnign with python 저자가 MIT License하에 직접 공개한 jupyter notebook을 우리말로 번역과 CodeOnWeb 실습 플랫폼에서 동작하도록 편집한 것입니다.
"""