
# url - https://codeonweb.com/entry/ea4cc4af-f8bb-484f-95b3-a348da93eb26

import keras
print(keras.__version__)

"""
합성곱 신경망(Convnet, Convolutional Neural Network) 시작하기

아주 단순한 합성곱 신경망 실제 예제를 한번 살펴 봅시다. 1강에서 연결 밀도가 빽빽한 신경망을 사용했었던(시험 정확도 97.8%였죠) MNIST 숫자 분류 문제를 합성곱 신경망(CNN)으로 적용할 것입니다. 사용할 합성곱이 아주 기초적이라 하더라도 정확도는 1강의 연결 밀도가 빽빽한 신경망 모델의 정확도 보다는 훨씬 높을 것입니다.

합성곱 신경망

아래의 6줄짜리 코드는 기초적인 합성곱 신경망이 어떻게 생겼는지 보여줍니다. Conv2D과 MaxPooling2D 계층의 스택이군요. 각각이 무엇을 하는지 잠시 후에 상세하게 살펴볼 겁니다. 중요한 점은 합성곱 신경망은 (image_height, image_width, image_channels)(batch 차원은 포함되지 않았습니다)과 같은 형태의 입력 텐서를 사용한다는 것입니다. 여기서는 MNIST 이미지의 형식인 (28, 28, 1) 크기의 입력을 처리하기 위한 합성곱 신경망을 구성하겠습니다. 이건 input_shape=(28, 28, 1) 인수를 첫 번째 계층에 넘겨주면 됩니다.
"""

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

"""
위에서 보시다시피 각 Conv2D, MaxPooling2D 계층의 출력은 (height, width, channels) 형태의 3차원 텐서 입니다. 너비와 높이 차원은 신경망으로 깊이 들어갈 수록 줄어드는 경향이 있습니다. 채널의 갯수는 Conv2D 계층으로 넘겨지는 첫 번째 인수로(여기서는 32 혹은 64) 조절할 수 있습니다.

다음 단계는 마지막 출력 텐서((3, 3, 64) 형태)를 이제 익숙한 연결 밀도가 빽빽한 분류기(classifier) 신경망, Dense 계층의 스택에 넣겠습니다. 분류기는 1차원인 벡터를 처리하는데 출력은 3차원의 텐서입니다. 그래서 우선은 3차원의 텐서를 1차원으로 평탄화(flatten)하고 두어 개의 Dense 계층을 추가합니다.
"""

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

"""
숫자가 10까지 종류이므로 10-way 분류를 할 것입니다. 10개 분류를 위한 마지막 계층은 10개의 출력과 소프트맥스 활성화를 사용하겠습니다. 이제 신경망은 다음과 같습니다:
"""

model.summary()

"""
보시다시피, (3, 3, 64) 출력은 (576,) 형태의 벡터로 평탄화되어서 두 개의 Dense 계층으로 들어갑니다.

MNIST 를 CNN을 이용해서 학습하자.

이제 합성곱 신경망을 MNIST 숫자에 대해 학습시킵니다. 1강의 MNIST 예제에서 사용한 코드를 많이 재사용할 겁니다:
"""

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)

"""
1강의 연결 밀도가 빽빽한 신경망의 시험 정확도는 97.8% 였는데 비하여, 기초적인 합성곱을 이용한 신경망의 시험의 정확도는 99.3% 입니다: (상대)에러율을 68%나 줄였습니다. 나쁘지 않네요!
"""
