
# url - https://codeonweb.com/entry/6fb77fab-1f69-4fb1-aec7-b75483d7e504

import keras
print(keras.__version__)

"""
합성곱 신경망이 학습한 것을 시각화하기

학습 표현은 추출하여서 사람이 읽을 수 있는 형식으로 제공하기가 어렵기 때문에, 딥러닝 모델은 "블랙 박스" 같다는 얘기를 합니다.
어떤 종류의 딥러닝 모델에 있어서 부분적으로는 옳은 말이지만, 합성곱 신경망에 있어서는 전혀 옳지 않은 말입니다. 합성곱 신경망에 의해 학습된 표현은 시각적 개념의 표현이기 때문에 일반적으로 시각화하기에 매우 적합합니다. 2013년부터 이러한 표현을 시각화하고 해석하기 위한 폭 넓은 일련의 기술들이 개발되어 왔습니다. 이러한 기술 모두를 둘러보지 못하겠지만 가장 사용하기 쉽고 유용한 기술 중 세 가지를 다뤄보도록 하겠습니다:

I. 중간 합성곱 신경망 출력 시각화("중간 활성화(intermediate activations)"): 이어지는 합성곱 계층이 입력을 어떻게 변형하는지 이해하고 개별 합성곱 필터가 가지는 의미에 대한 개념을 잡는 데에 유용합니다.
II. 합성곱 필터 시각화: 합성곱 신경망의 각 필터가 어떤 시각 패턴이나 개념을 받아들일 수 있는지 이해하는 데에 유용합니다.
III. 이미지 내의 범주 활성화의 히트맵 시각화: 이미지 내의 어떤 부분이 주어진 범주에 속하는 것으로 식별되는지를 이해하는 데에 유용하여 이미지 내의 객체를 지역화할 수 있도록 합니다.
I. 중간 활성화 시각화하기

첫 번째 방법, 활성화 시각화에 대해서는 두 절 앞의 고양이 vs. 개 분류 문제에 대해 백지에서부터 훈련한 작은 합성곱 신경망을 사용하겠습니다. 다음의 두 방법에 대해서는 앞 절에서 소개한 VGG16 모델을 사용하도록 하겠습니다.

중간 활성화 시각화하기는 주어진 입력에 대해 신경망 내의 다양한 합성곱과 풀링 계층이 출력하는 특징 맵을 표시합니다(계층의 출력은 종종 "활성화", 활성화 함수의 출력으로 불립니다). 이를 통해 입력이 신경망에 의해 학습된 서로 다른 필터에 어떻게 분석되는지 볼 수 있습니다. 시각화하고자 하는 특징 맵은 3개의 차원이 있습니다: 폭, 높이, 깊이(채널들입니다). 각 채널은 상대적으로 독립적인 특징을 인코딩하므로 이런 특징 맵을 시각화하는 적절한 방법은 2차원 이미지와 같이 각 채널의 내용을 독립적으로 그래프화하는 것입니다. 그럼 케라스 5장에서 저장한 모델을 불러오는 것으로 시작해 봅시다:
"""

from keras.models import load_model

path = keras.utils.get_file(
    'cats_and_dogs_small_2.h5',
    'http://datasets.lablup.ai/public/tutorials/cats_and_dogs_small_2.h5')
model = load_model(path)
model.summary()  # As a reminder.


"""
사용할 입력 이미지는 다음과 같이 신경망이 훈련된 이미지의 일부분이 아닌 고양이의 사진 입니다:
"""

import os

dataset_file = keras.utils.get_file(
    'cats-and-dogs.zip',
    'http://datasets.lablup.ai/public/kaggle/cats-and-dogs-small/dataset.zip',
    extract=True)
dataset_dir = os.path.dirname(dataset_file)
img_path = os.path.join(dataset_dir, 'test/cats/cat.1700.jpg')

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.

# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

"""
사진을 표시해 봅시다
"""

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

"""
보고자 하는 특징 맵을 추출하기 위해 이미지 batch를 입력으로 받아 모든 합성곱과 풀링 계층의 활성화를 출력하는 Keras 모델을 만들어 보겠습니다. 이 작업을 위해 Keras의 클래스 Model을 사용하겠습니다. Model은 두 개의 인수를 사용하여 인스턴스화됩니다: 입력 텐서(혹은 입력 텐서의 리스트)와 출력 텐서(혹은 출력 텐서의 리스트). 결과 클래스는 이미 익숙하실 Sequential 모델과 같이 지정된 입력을 지정된 출력으로 대응시키는 Keras 모델 입니다. Model 클래스를 구분하는 차이는 Sequential과 달리 모델에 대해 다중 출력을 허용한다는 점입니다. Model 클래스에 대한 좀 더 상세한 정보는 케라스 강의 7장 1절을 참조하세요.
"""

from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

"""
이미지 입력이 주어지면 이 모델은 원래 모델의 계층 활성화 값이 반환됩니다. 이 책에서 다중 출력 모델은 이게 처음일 겁니다: 지금까지 살펴 본 모든 모델은 정확히 하나의 입력에 하나의 출력만 있었으니까요. 일반적으로 모델은 어떠한 갯수의 입력과 출력도 다 가질 수 있습니다. 이 경우는 하나의 입력에 8개의 출력을 가지고 있고 하나의 계층 활성화 마다 하나의 출력이 있습니다.
"""

# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

"""
예를 들어, 고양이 이미지 입력에 대한 첫 번째 합성곱 계층 활성화는 다음과 같습니다:
"""

first_layer_activation = activations[0]
print(first_layer_activation.shape)

"""
32개 채널의 148x148 특징 맵이네요. 그럼 세 번째 채널를 한번 시각화해 보겠습니다:
"""

import matplotlib.pyplot as plt
plt.close() # To clear previous figure

plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()


"""
이 채널은 비스듬한 가장자리를 감지하도록 인코딩되어 있네요. 그럼 30번째 채널을 시도해 보겠습니다; 다만, 합성곱 계층에 의해 학습된 개별 필터는 결정되어 있는 게 아니기 때문에 직접 실행해 보면 다를 수 있다는 점을 유의하세요.
"""

plt.close() # To clear previous figure

plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

"""
이건 "밝은 녹색 점"을 감지하는 것 같네요, 고양이 눈을 인코딩하는 데에 유용하겠죠. 이쯤에서 신경망의 모든 활성화를 전부 시각화해서 표시해 볼까요. 8개의 각 활성화 맵의 모든 채널을 추출하여 그 결과를 채널 별로 나란히 쌓아 하나의 커다란 이미지 텐서로 쌓아 표시하겠습니다.
"""

import keras

# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

plt.close() # To clear previous figure

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

for fig in range(8): # To clear previous figure
    plt.close(fig)   

"""
주목할 만한 점이 몇 가지 있죠:

첫 번째 계층은 다양한 가장자리 감지기를 모아 놓은 것 같네요. 이 단계에서, 활성화는 원래의 사진이 제공하는 거의 대부분의 정보를 여전히 간직하고 있습니다.
좀 더 높은 계층으로 갈 수록, 활성화는 더 추상화되어 시각적으로 해석할 여지가 적어집니다. "고양이 눈"이나 "고양이 귀"와 같은 좀 더 높은 수준의 개념을 인코딩하기 시작합니다. 더 높은 계층의 표현은 이미지의 시각적 내용에 관한 정보는 더 적게 지니는 대신 이미지의 범주에 관련된 정보는 좀 더 많이 지닙니다.
활성화의 희박함은 계층이 깊어짐에 따라 증가합니다: 첫 번째 계층에서 모든 필터는 입력 이미지에 의해 활성화되었지만 이어지는 계층에서는 점점 더 많은 필터가 비어있습니다. 이것은 필터에 의해 인코딩되는 패턴이 입력 이미지에서 찾을 수 없다는 의미 입니다.
방금 깊은 신경망에 의해 학습된 표현의 아주 중요한 보편적 특성을 증명하였습니다: 계층에 의해 추출된 특징은 계층의 깊이가 증가함에 따라 더 추상화 됩니다. 더 깊은 계층의 활성화는 보여지는 구체적인 입력의 정보는 더 적게 지니고, 목표(여기서는 이미지의 범주: 고양이 혹은 개)에 대한 정보는 더 많이 지닙니다. 깊은 신경망은 들어오는 원본 데이터(여기서는 RBG 사진)에 대해 관련성이 떨어지는 정보(예를 들어 이미지의 구체적인 시각적 외관)는 걸러버리는 반면, 유용한 정보(예를 들어 이미지의 범주)는 증폭하고 정제함으로써, 정보를 증류하는 파이프라인으로 효과적으로 작동합니다.

이런 과정은 인간과 동물이 세상을 인지하는 방식과 유사합니다: 어떤 장면을 몇 초간 관찰한 다음, 인간은 거기에 어떤 추상적 객체(예를 들어 자전거, 나무 등)가 있었는지는 기억할 수 있지만 그 객체의 구체적인 외관은 기억하지 못합니다. 사실, 지금 바로 일반적인 자전거를 마음 속으로 그리려고 시도해 본다면, 평생 동안 몇천번이나 자전거를 보았음에도, 조금 비슷하게나마도 그리지 못 할 겁니다. 지금 도전해 보세요: 이 효과는 절대적으로 사실입니다. 두뇌는 시각적 입력을 완전히 추상화하여 높은 수준의 시각적 개념으로 변형하는 한편, 관련성 낮은 시각적 세부 사항은 걸러버리고 학습함으로써, 주변 사물이 실제로 어떻게 보이는지 기억하는 것을 굉장히 어렵게 합니다.

합성곱 신경망 필터를 시각화하기

합성곱 신경망으로 학습한 필터를 점검하는 다른 하나의 쉬운 방법은 각 필터가 반응하고자 하는 시각적 패턴을 표시하는 것입니다. 이 작업은 입력 공간의 기울기 상승을 통해 수행할 수 있습니다: 비어 있는 입력 이미지부터 시작하여 특정 필터의 반응을 최대화하도록 기울기 하강을 합성곱 신경망의 입력 이미지의 값에 적용하는 겁니다. 결과 입력 이미지는 선택한 필터가 최대로 반응하는 이미지입니다.

과정은 간단합니다: 주어진 합성곱 계층의 주어진 필터 값을 최대화하는 손실 함수를 만든 다음, 활성화 값이 최대화되도록 입력 이미지의 값을 조정하게 확률적 기울기 하강을 사용합니다. 예를 들어, ImageNet에서 사전 훈련된 VGG16 신경망의 "block3_conv1" 계층의 필터 0의 활성화에 대한 손실은 다음과 같습니다:
"""

from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

""" 
기울기 하강을 구현하기 위해, 모델의 입력에 대한 손실의 기울기가 필요합니다. 이를 위해, Keras의 backend 모듈에 패키지로 제공된 gradients 함수를 사용하겠습니다:
"""

# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads = K.gradients(loss, model.input)[0]

"""
기울기 하강 과정을 부드럽게 진행하는 데에 사용하는 알기 쉽지는 않은 트릭은 기울기 텐서를 L2 norm(텐서 내의 값들의 제곱의 평균에 대한 제곱근)으로 나누어 표준화 하는 것입니다. 이를 통해 입력 이미지에 대한 업데이트의 크기가 항상 같은 범위 내에 있음을 보장할 수 있습니다.
"""

# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

"""
이제 주어진 입력 이미지의 손실 텐서와 기울기 텐서의 값을 계산할 방법이 필요합니다. 이를 위한 Keras 백엔드 함수를 정의할 수 있습니다: iterate는 Numpy 텐서(크기가 1인 텐서의 리스트로서)를 받아서 두 개의 Numpy 텐서의 리스트(손실 값과 기울기 값)를 반환하는 함수입니다.
"""

iterate = K.function([model.input], [loss, grads])

# Let's test it:
import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

"""
여기서 확률적 기울기 하강을 수행할 파이썬 루프를 정의할 수 있습니다:
"""

# We start from a gray image with some noise
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# Run gradient ascent for 40 steps
step = 1.  # this is the magnitude of each gradient update
for i in range(40):
    # Compute the loss value and gradient value
    loss_value, grads_value = iterate([input_img_data])
    # Here we adjust the input image in the direction that maximizes the loss
    input_img_data += grads_value * step

"""
결과 이미지 텐서는 (1, 150, 150, 3) 형태의 [0, 255] 내의 정수가 아닌 값을 가지는 부동소수점 텐서입니다. 따라서 이 텐서를 표시 가능한 이미지로 변환하기 위한 후처리가 필요합니다. 다음과 같은 직접적인 도구 함수를 사용하면 됩니다:
"""

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

"""
이제 모든 조각을 다 모았으니, 계층 이름과 필터 인덱스를 입력으로 받아 지정된 필터의 활성화를 최대화하는 패턴을 표현하는 효과적인 이미지 텐서를 반환하는 파이썬 함수로 만들어 봅시다:
"""

def generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)

plt.close() # To clear previous figure

plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

"""
block3_conv1 계층의 필터 0은 물방울 무늬 패턴에 반응하는 듯 보입니다.

이제 재밌는 부분입니다: 각 계층의 개별 필터를 시각화할 수 있습니다. 단순화를 위해, 각 계층의 첫 64개 필터만 보고, 각 합성곱 블록의(block1_conv1, block2_conv1, block3_conv1, block4_conv1) 첫 번째 계층만 보겠습니다. 출력을 각 필터 패턴 사이가 검은 여백으로 구분된 64x64 필터 패턴의 8x8 격자로 정렬하겠습니다.
"""

plt.close() # To clear previous figure

for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    size = 64
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)

plt.show()

"""
이런 필터 시각화는 합성곱 신경망이 세상을 어떻게 보는지에 대해 많은 것을 얘기해 줍니다: 합성곱 신경망의 각 계층은 단순히 필터의 모음을 학습하여 입력을 필터의 조합으로 표현합니다. 이 방식은 푸리에 변환이 신호를 코사인 함수의 집합으로 분해하는 방식과 유사합니다. 이 합성곱 신경망 필터 집합의 필터는 모델의 상단으로 갈수록 복잡해지고 정제됩니다:

모델의 첫 번째 계층의 필터(block1_conv1)는 간단한 방향 가장자리와 색상(어떤 경우에는 색상의 가장자리)을 인코딩합니다.
block2_conv1의 필터는 가장자리와 색상의 조합으로 만들어진 간단한 질감을 인코딩합니다.
좀 더 상단 계층의 필터는 자연 이미지의 질감을 흉내내기 시작합니다: 깃털, 눈, 잎 등등.
범주 활성화의 히트맵을 시각화하기

주어진 이미지의 어느 부분이 합성곱 신경망의 최종 분류 결정을 이끌었는지 이해하는 데에 유용한 시각화 기술을 하나 더 도입해 보겠습니다. 이 기술은 특히 분류 착오가 있는 합성곱 신경망의 결정 과정을 "디버깅"하는 데에 유용합니다. 이 기술은 이미지 내의 특정 개체를 찾는 것도 가능하게 해줍니다.

이런 기술을 일반적으로 "범주 활성화 맵(Class Activation Map, CAM)" 시각화라 하며, 입력 이미지 상에서 "범주 활성화"의 히트맵을 산출합니다. "범주 활성화" 히트맵은 특정 출력 범주와 입력 이미지의 각 위치가 얼마나 관련이 있는지 계산된 점수의 2차원 그리드로서 해당 범주에 대해 각 위치가 얼마나 중요한지를 나타냅니다. 예를 들어, "고양이 vs. 개" 합성곱 신경망에 이미지가 주어지면 범주 활성화 맵 시각화는 이미지의 각 부분이 얼마나 고양이스러운지를 나타내는 "고양이" 범주의 히트맵을 생성하고 마찬가지로 각 부분이 얼마나 개스러운지를 나타내는 "개" 범주의 히트맵을 생성합니다.

사용할 상세한 구현은 다음 논문에 설명된 것 중 하나를 사용하도록 하겠습니다: Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via Gradient-based Localization 아주 간단합니다: 주어진 입력 이미지에 대한 합성곱 계층의 출력 히트맵을 받아서 채널에 대한 범주의 기울기로 각 채널을 계량합니다. 직관적으로는, 이 트릭을 이해하는 한 방법은 "얼마나 강하게 입력 이미지가 서로 다른 채널을 활성화하는지"에 대한 공간적인 맵을 "얼마나 각 채널이 범주에 대해 중요한지"로 가중치를 부여하여 "얼마나 강하게 입력 이미지가 해당 범주를 활성화하는지"에 대한 공간적인 맵을 결과로 보여줍니다.

사전 훈련된 VGG16 신경망을 다시 사용하여 이 기술은 실제로 써보겠습니다:
"""

from keras.applications.vgg16 import VGG16

K.clear_session()

# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = VGG16(weights='imagenet')

"""
사바나를 거닐고 있는 어미 코끼리와 새끼 코끼리의 이미지를 한번 살펴봅시다(Creative Commons license 하의 이미지입니다

이 이미지를 VGG16 모델이 읽을 수 있도록 변환해 봅시다: 모델은 keras.applications.vgg16.preprocess_input의 도구 함수에 패키지된 몇 개의 규칙에 따라 전처리된 224x224 크기의 이미지에 훈련되었습니다. 따라서 이미지를 불러와서 224x224 크기로 바꾸고 Numpy float32 텐서로 변환한 후, 전처리 규칙을 적용하겠습니다.
"""

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Load the image
img_path = keras.utils.get_file(
    'elephants.jpg',
    'http://datasets.lablup.ai/public/tutorials/elephants.jpg')

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

"""
이 이미지에 가장 많이 예측되는 3개의 범주는 다음과 같습니다:

아프리칸 코끼리 (92.5% 확률)
터스커 (7% 확률)
인도 코끼리 (0.4% 확률)
따라서 신경망은 명확하지 않은 정도의 아프리카 코끼리를 포함하고 있다고 인지합니다. 최대로 활성화된 예측 벡터의 엔트리는 인덱스 386의 "아프리카 코끼리" 범주와 관련됩니다:
"""

np.argmax(preds[0])

"""
이미지의 어느 부분이 가장 "아프리카 코끼리" 같은지 시각화하기 위해, Grad-CAM 과정을 설정해 봅시다
"""

# This is the "african elephant" entry in the prediction vector
african_elephant_output = model.output[:, 386]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

"""
시각화하기 위해, 히트맵을 0과 1 사이 값으로 표준화 하겠습니다:
"""

for fig in range(4):
    plt.close() # To clear previous figure

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

"""
최종적으로, OpenCV를 사용하여 방금 얻은 히트맵을 원본 이미지와 중첩시키는 이미지를 생성해 봅시다:
"""

import cv2

# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)

""" 
이런 시각화 기술은 두 가지 중요한 질문에 대해 대답합니다:

신경망은 왜 이 이미지가 아프리카 코끼리를 포함하고 있다고 생각했습니까?
사진 내의 어디에 아프리카 코끼리가 있습니까?
특히 새끼 코끼리의 귀가 활성화되어 있는 점이 주목할 만 합니다: 아마도 이 부분을 통해 신경망이 아프리카 코끼리와 인도 코끼리를 구별할 수 있을 겁니다.

이 강의는 Keras를 만든 Francois Chollet가 쓴 딥러닝 입문서 "Deep Learning with Python (Manning Publications)"의 5장 4절의 코드 샘플이며 저자가 직접 공개한 jupyter notebook을 우리말로 번역하고 실습 플랫폼에 연결한 것입니다. 여기에는 소스 코드와 주석만 있고, 책에 상세한 설명과 그림 등을 포함한 풍부한 내용이 있습니다!
"""