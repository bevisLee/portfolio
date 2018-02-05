
# url - https://codeonweb.com/entry/9a579d8d-d14e-474c-b7fe-19237cffe9e0

import keras
print(keras.__version__)

"""
신경망 스타일 이식 (Neural style transfer)

딥 드림 외에 딥 러닝을 이용한 이미지 변형의 또 하나의 주요 개발품은 Leon Gatys 등에 의해 2015년 여름에 소개된 신경망 스타일 이식(neural style transfer)입니다. 신경망 스타일 이식 알고리즘은 최초 소개된 이후에 많은 세부 조정을 거쳐 다양한 변형을 낳았는데 이 중에는 바이럴 스마트폰 앱인 Prisma도 있습니다. 간단하게 하기 위해, 이 절에서는 원본 논문에 기술된 내용에 집중하도록 하겠습니다.

신경망 스타일 이식은 목표 이미지의 "컨텐츠"을 유지하면서 참조 이미지의 "스타일"을 목표 이미지에 적용하는 것으로 구성됩니다:

file
"스타일"이 의미하는 바는 본질적으로 다양한 범위에서 이미지의 질감, 색상, 그리고 시각적 패턴인 반면, "컨텐츠"는 좀 더 고수준의 이미지의 육안적 구조를 의미합니다. 예를 들어 파란색과 노란색의 붓 터치가 위 예제에 쓰인 반 고흐의 "별이 빛나는 밤"의 "스타일"로 여겨지는 한편, 튀빙겐(Tuebingen)의 사진은 "컨텐츠"로 여겨집니다.

질감의 생성과 단단히 연결되어 있는 스타일 이식의 아이디어는 2015년에 나온 신경망 스타일 이식 이전에도 이미지 처리 분야에서 긴 역사를 가지고 있습니다. 하지만 밝혀진 바와 같이 딥 러닝 기반의 스타일 이식 구현은 이전의 고전적인 컴퓨터 비전 기술로 달성할 수 있었던 수준과는 비교가 안 되는 뛰어난 결과물을 제공하여서 컴퓨터 비전 분야의 창의적 응용의 깜짝 놀랄 만한 르네상스를 촉발시켰습니다.

스타일 이식을 구현하는 핵심 개념은 모든 딥 러닝 알고리즘의 핵심 아이디어와 동일합니다: 달성하고자 하는 것을 지정하기 위해 손실 함수를 정의하고 이 손실을 최소화합니다. 여기서 달성하고자 하는 것은 잘 알고 있죠: 원본 이미지의 "컨텐츠"를 유지하면서 참조 이미지의 "스타일"을 차용하고자 합니다. 수학적으로 컨텐츠와 스타일을 정의할 수 있다면 손실을 최소화 하기 위한 적절한 손실 함수는 다음과 같습니다:

loss=distance(style(reference_image)−style(generated_image))+ distance(content(original_image)−content(generated_image))
loss=distance(style(reference_image)−style(generated_image))+ distance(content(original_image)−content(generated_image))

distance는 L2 norm과 같은 norm 함수이고, content는 이미지를 받아서 "컨텐츠"의 표현을 계산하는 함수이며 style는 이미지를 받아서 "스타일"의 표현을 계산하는 함수입니다.

이 손실을 최소화하는 작업은 style(generated_image)을 style(reference_image)에 가깝게 하는 한편, content(generated_image)을 content(generated_image)에 가깝게 하여 정의한 바와 같이 스타일 이식을 달성할 수 있도록 합니다.

Gatys 등의 근본적인 관찰에 따르면 깊은 합성곱 신경망은 style과 content 함수를 수학적으로 정의할 정확한 방법을 제공해 줍니다. 어떻게 하는지 한번 봅시다.

컨텐츠 손실

이젠 잘 아시다시피, 신경망의 앞쪽 계층의 활성화는 이미지의 국부적 정보를 포함하는 반면, 상단 계층의 활성화는 점점 더 전체적이고 추상화된 정보를 포함합니다. 서로 다른 방식으로 기술된 합성곱 신경망의 서로 다른 계층의 활성화는 서로 다른 공간적 규모에서 이미지의 컨텐츠를 분해하여 제공합니다. 따라서 이미지의 "컨텐츠", 즉 좀 더 전체적이고 좀 더 추상화된 정보를 합성곱 신경망의 최상위 계층이 표현할 것이라고 기대할 수 있습니다.

그래서 컨텐츠 손실의 유력한 후보로 사전에 훈련된 합성곱 신경망이 고려될 수 있고 목표 이미지에 대해 계산된 최상위 계층의 활성화와 생성된 이미지에 대해 계산된 동일 계층의 활성화 간의 L2 norm이 손실로 정의됩니다. 이 점이 합성곱 신경망의 최상위 계층에서 보여지는 바대로, 생성된 이미지가 원본 목표 이미지와 "유사하게 보일" 거라는 것을 보장합니다. 합성곱 신경망의 최상위 계층 보는 것이 실제로 입력 이미지의 "컨텐츠"라고 가정하면, 이미지의 컨텐츠를 유지하는 방법으로 작동할 겁니다.

스타일 손실

컨텐츠 손실이 최상위 계층 하나만을 사용하는 반면에, Gatys 등의 논문에 정의된 스타일 손실은 합성곱 신경망의 계층 여러 개를 사용합니다: 단 하나의 범위에서가 아닌 모든 공간적 범위에서 합성곱 신경망에 의해 추출된 스타일 참조 이미지의 표현을 포착해 내는 것을 목표로 합니다.

스타일 손실에 대해, Gatys 등의 논문에서는 계층 활성화의 "Gram 행렬", 즉 주어진 계층의 특징 맵 간의 내적을 사용합니다. 이 내적은 계층의 특징들 간의 상관관계의 맵을 표현하는 것으로 이해될 수 있습니다. 이 특징 상관관계는 경험적으로 특정 공간적 범위에서 발견되는 질감의 표현과 부합되는 해당 공간적 범위의 패턴의 통계치를 포착합니다.

따라서 스타일 손실은 스타일 참조 이미지와 생성된 이미지를 가로질러 서로 다른 계층의 활성화의 유사한 내부 상관관계를 유지하는 것을 목표로 합니다. 결과적으로, 이 점이 서로 다른 공간적 범위에서 발견되는 질감이 스타일 참조 이미지와 생성된 이미지를 가로질러 유사하게 보이도록 보장해 줍니다.

요약

요약하자면, 다음과 같이 사전 훈련된 합성곱 신경망을 손실을 정의하는 데에 사용할 수 있습니다:

목표 컨텐츠 이미지와 생성된 이미지 사이의 고수준 계층 활성화를 비슷하게 유지함으로써 컨텐츠를 보존할 수 있습니다. 합성곱 신경망은 목표 이미지와 생성된 이미지 양 쪽이 모두 "같은 것들을 포함하고 있다"고 "보아야" 합니다.
저수준 계층과 고수준 계층 모두의 활성화 내의 상관관계를 비슷하게 유지함으로써 스타일을 보존할 수 있습니다. 실제로, 특징 상관관계는 질감을 포착합니다: 생성된 이미지와 스타일 참조 이미지는 서로 다른 공간적 범위에서 동일한 질감을 공유하여야 합니다.
그럼 Keras로 구현한 2015년 원래의 신경망 스타일 이식 알고리즘을 한번 살펴봅시다. 보시다시피, 앞 절에서 만들어 본 바 있는 딥 드림의 구현과 많은 유사점을 공유하고 있습니다.

Keras의 신경망 스타일 이식

신경망 스타일 이식은 어떤 사전 훈련된 합성곱 신경망을 써도 구현할 수 있습니다. 여기서는 Gatys 등이 논문에서 사용했던 VGG19 신경망을 써보도록 하겠습니다. VGG19는 5장에서 소개한 바 있는 VGG16에 3개의 합성곱 계층이 추가된 간단한 변종입니다.

전체 작업은 다음과 같습니다:

스타일 참조 이미지, 목표 이미지 그리고 생성 이미지의 VGG19 계층 활성화를 동시에 계산할 신경망을 설정합니다.
이 세 가지 이미지에 대해 계산된 계층 활성화를 사용하여 위에서 설명한 것 처럼 스타일 이식을 위해 최소화될 손실 함수를 정의합니다.
이 손실 함수를 최소화할 기울기 하강 작업을 설정합니다.
고려해야 할 두 가지 이미지, 스타일 참조 이미지와 목표 이미지의 경로를 정의하는 것으로 시작해 봅시다. 처리되는 모든 이미지가 비슷한 크기를 공유하도록 하기 위해(이미지 크기가 너무 차이가 나면 스타일 이식이 더 어려울 수 있습니다), 이따가 모든 이미지의 높이를 동일하게 400px로 조정할 겁니다.
"""

from keras.preprocessing.image import load_img, img_to_array

# This is the path to the image you want to transform.
target_image_path = keras.utils.get_file(
    'girl.jpg',
    'http://datasets.lablup.ai/public/tutorials/girl-at-park.jpg')
# This is the path to the style image.
style_reference_image_path = keras.utils.get_file(
    'gogh.jpg',
    'http://datasets.lablup.ai/public/tutorials/gogh-selfportrait1889.jpg')

# Dimensions of the generated picture.
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

"""
VGG19 합성곱 신경망에 이미지를 넣고 빼기 위해 이미지를 불러 오고 전처리하고 후처리하는 몇 가지 보조 함수가 필요합니다:
"""

import numpy as np
from keras.applications import vgg19

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

"""
VGG19 신경망을 설정해 봅시다. 다음과 같은 3가지 이미지의 batch를 입력으로 받습니다: 스타일 참조 이미지와 목표 이미지 그리고 생성된 이미지를 포함할 위치표시자(placeholder)입니다. 위치표시자는 간단히 말하자면 상징적 텐서로 그 값이 Numpy 배열을 통해 외부로 표시됩니다. 스타일 참조 이미지와 목표 이미지는 정적이므로 K.constant를 사용하여 정의되는 반면, 생성 이미지의 위치표시자에 포함되는 값은 시간이 지남에 따라 변하게 됩니다.
"""

from keras import backend as K

target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# This placeholder will contain our generated image
combination_image = K.placeholder((1, img_height, img_width, 3))

# We combine the 3 images into a single batch
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# We build the VGG19 network with our batch of 3 images as input.
# The model will be loaded with pre-trained ImageNet weights.
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

"""
VGG19 합성곱 신경망의 최상위 계층이 목표 이미지와 생성된 이미지를 비슷하게 볼 수 있도록 하는 컨텐츠 손실을 정의해 봅시다:
"""

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

"""
이제 스타일 손실 차례입니다. 원본 특징 행렬에서 발견되는 상관관계의 맵인 입력 행렬의 Gram 행렬을 계산하는 보조함수를 이용해 봅시다.
"""

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

"""
이 두 가지의 손실에 세번째 손실, "총 변화 손실(total variation loss)"을 추가합니다. 이 손실은 생성된 이미지의 공간적인 연속성을 장려하여, 결과가 과도하게 픽셀화되는 것을 방지합니다. 정규화 손실로 해석할 수 있습니다.
"""

def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

"""
최소화할 손실은 이 세 가지 손실의 가중 평균 입니다. 컨텐츠 손실을 계산하기 위해, 최상위 계층 하나, block5_conv2 계층만을 사용하는 반면, 스타일 손실은 저수준과 고수준 계층 모두를 포괄하는 계층의 리스트를 사용합니다. 마지막으로 총 변동 손실을 추가합니다.

사용하는 스타일 참조 이미지와 컨텐츠 이미지에 따라, content_weight 계수, 즉 총 손실에 대한 컨텐츠 손실의 기여도를 적절하게 조정할 수 있습니다. 더 높은 content_weight는 목표 컨텐츠가 생성된 이미지에서 더 잘 인식된다는 뜻입니다.
"""

# Dict mapping layer names to activation tensors
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Name of layer used for content loss
content_layer = 'block5_conv2'
# Name of layers used for style loss
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# Weights in the weighted average of the loss components
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Define the loss by adding all components to a `loss` variable
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

"""
마지막으로, 기울기 하강 작업을 설정합시다. 원래의 Gatys 등의 논문에서 최적화가 L-BFGS 알고리즘을 사용하여 수행되었으므로 여기서도 같은 알고리즘을 써보겠습니다. 앞 절의 딥 드림 예제와의 핵심적인 차이가 바로 이것입니다. L-BFGS 알고리즘은 SciPy에 패키지로 제공됩니다. 하지만, SciPy 구현에는 두 가지 작은 제한 사항이 있습니다:

손실 함수의 값과 기울기의 값을 두 개의 분리된 함수로 넘겨주기를 요구합니다.
3차원 이미지 배열을 사용하고 있는데, 이 구현은 평평한 벡터에만 적용 가능합니다.
손실 함수의 값과 기울기의 값 사이에는 너무 정황한 계산을 필요로 하기 때문에 이 둘을 별도로 계산한는 것은 너무나 비효율적일 겁니다. 이 둘을 같이 계산하는 것에 비하면 거의 두 배는 느릴 겁니다. 이를 우회하기 위해 손실 값과 기울기 값을 한번에 계산할 Evaluator라는 Python 클래스를 설정하여서 처음 호출될 때 손실 값을 반환하고, 다음 호출을 위해 기울기를 캐쉬하도록 합시다.
"""

# Get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)[0]

# Function to fetch the values of the current loss and the current gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

"""
끝으로, SciPy의 L-BFGS 알고리즘을 사용하여 기울기 상승 작업을 실행하여 알고리즘의 각 반복수행 마다 현재 생성된 이미지를 저장합니다(여기서는 한 번의 수행은 20 단계의 기울기 상승을 의미합니다):
"""

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'style_transfer_result'
iterations = 20

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss.
# This is our initial state: the target image.
# Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # Save current generated image
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

from matplotlib import pyplot as plt

# Content image
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# Style image
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# Generate image
plt.imshow(img)
plt.show()

"""
이 기술로 얻은 것은 단지 이미지 질감 변경 혹은 질감 이식의 한 형태에 불과함을 명심하세요. 이 작업은 질감이 뚜렷하고 자기 유사성이 높은 스타일 참조 이미지와 인식되기 위해 높은 수준의 세부 사항이 요구되지 않는 컨텐츠 목표 이미지에 적합합니다. 일반적으로는 "한 초상화의 스타일을 다른 초상화로 이식하기"와 같은 추상적인 솜씨를 부리지는 못할 겁니다. 이 알고리즘은 인공지능보다는 고전적인 신호 처리에 가깝기 때문에, 마술 같은 걸 기대하지 마세요!

또한, 이 스타일 이식 알고리즘은 꽤 느리다는 점을 꼭 유념하세요. 다만, 이 설정에 의한 변형 작업은 충분히 단순하기 때문에 작고 빠른 피드포워드(feedforward) 합성곱 신경망으로도 학습시킬 수 있습니다, 사용 가능한 훈련 테이터가 충분하다면 말이죠. 따라서 위의 방법을 사용하여 하나의 고정된 스타일 참조 이미지에 대한 입-출력 훈련 예제를 생성하는 데에 한번 많은 계산 사이클을 들인 다음, 이 스타일에 특정한 변형을 학습할 작은 합성곱 신경망을 훈련한다면 빠른 스타일 이식을 만들 수 있습니다. 이 작업이 한번 수행되면, 주어진 이미지에 스타일을 적용하는 작업은 즉각적입니다: 이건 이 작업 합성곱 신경망에 대한 전방 전달에 불과하니까요.

시사점

스타일 이식은 목표 이미지의 "컨텐츠"를 유지하는 한편, 참조 이미지의 "스타일" 역시 포착한 새로운 이미지를 생성하는 것으로 구성됩니다.
"컨텐츠"는 합성곱 신경망의 고수준 활성화에 의해 포착될 수 있습니다.
"스타일"은 합성곱 신경망의 서로 다른 각각의 계층 내부의 활성화 간의 상관 관계에 의해 포착될 수 있습니다.
따라서 딥 러닝은 스타일 이식을 사전 훈련된 합성곱 신경망으로 정의된 손실의 최적화 작업으로 기술되도록 해줍니다.
이런 기본적인 아이디어에서 출발하여 많은 변형과 세부 조정이 가능합니다!
"""