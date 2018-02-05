### Tensorflow 도입
## 학습 목표

# 다음 개념에 중점을 둔 TensorFlow 프로그래밍 모델의 기초를 배웁니다.
 # 텐서 (tensors)
 # 작업 (operations)
 # 그래프 (graphs)
 # 세션들 (sessions)
# 기본 그래프를 만드는 간단한 TensorFlow 프로그램과 그래프를 실행하는 세션 만들기
# TensorFlow 프로그래밍 모델은 사용자가 만난 다른 프로그래밍 모델과 다를 수 있습니다. 그러므로 꼭 차근차근 열람하시기 바랍니다.

# 개념 개요

# TensorFlow는 임의의 차원의 배열인 텐서 (tensors)에서 유래한 이름입니다. TensorFlow를 사용하면 아주 많은 수의 차원을 가진 텐서로 작업 할 수 있습니다. 즉, 다음과 같은 평범한 차원에서 많은 작업이 이루어질 것입니다.

 # 스칼라는 0-d 배열 (0차 텐서)입니다. 예를 들어, "Howdy"또는 5
 # 벡터는 1차원 배열 (1차 텐서)입니다. 예를 들어 [2, 3, 5, 7, 11] 또는 [5]
 # 행렬은 2차원 배열 (2차 텐서)입니다. 예를 들어, [[3.1, 8.2, 5.9] [4.3, -2.7, 6.5]]
# TensorFlow 작업(operations)은 텐서를 만들고, 파괴하고 조작합니다. 일반적인 TensorFlow 프로그램의 코드는 대부분 작업입니다.

# TensorFlow 그래프graph (계산 그래프(computational graph) 또는 데이터 흐름 그래프 (dataflow graph) 라고도 함)는 그래프 데이터 구조입니다. 많은 TensorFlow 프로그램은 하나의 그래프로 구성되지만 TensorFlow 프로그램은 선택적으로 여러 개의 그래프를 만들 수 있습니다. 그래프의 각 노드는 연산에 해당합니다. 그래프의 연결선은 텐서입니다. 그래프를 따라 텐서가 흐르고 각 노드에서 작업에 의해 조작됩니다. 한 연산의 출력 텐서는 종종 후속 연산의 입력 텐서가 됩니다. TensorFlow는 게으른 실행 모델(lazy execution model) 을 구현합니다. 즉, 노드는 관련 노드의 필요에 따라 필요할 때만 계산됩니다.

# 텐서는 그래프에 상수 (constants) 또는 변수 (variables) 로 저장할 수 있습니다. 짐작할 수 있듯이 상수는 값을 변경할 수 없는 텐서를 담고, 변수는 값을 변경할 수 있는 텐서를 담습니다. 그러나 여러분이 짐작하지 못했던 것은 상수와 변수가 그래프에서 더 많은 작업일 뿐이라는 것입니다. 상수는 항상 동일한 텐서 값을 반환하는 작업입니다. 변수는 할당 된 텐서를 반환하는 작업입니다.

# 상수를 정의하려면 tf.constant 연산자를 사용하고 해당 값을 전달하세요. 다음은 예입니다.
import tensorflow as tf

x = tf.constant([5.2])
print(x)

# 마찬가지로 다음과 같은 변수를 만들 수 있습니다.
y = tf.Variable([5])
print(y)

# 또는 변수를 먼저 생성하고 나중에 이와 같이 값을 지정할 수 있습니다 (항상 기본값을 지정해야 함).
y = tf.Variable([0])
y = y.assign([5])
print(y)

# 일부 상수나 변수를 정의한 후에는 tf.add와 같은 다른 작업과 결합 할 수 있습니다. tf.add 작업을 실행할 때 tf.constant 또는 tf.Variable 연산을 호출하여 값을 얻은 다음 해당 값의 합계로 새로운 텐서를 반환합니다.

# tf.Variables 사용과의 중요한 차이점은 tf.global_variables_initializer 연산을 생성하고 세션을 시작할 때 이를 호출하여 명시 적으로 초기화 해야한다는 것입니다.

# 그래프는 TensorFlow 세션 내에서 실행되어야합니다. 세션(session) 은 실행되는 그래프의 모든 상태를 유지합니다. 또한 세션은 (프로그램이 일부 분산 계산 프레임 워크에서 실행된다고 가정할 경우) 여러 시스템에 그래프 실행을 분산시킬 수 있습니다.

# 따라서 TensorFlow 프로그래밍은 기본적으로 다음을 포함합니다.

# 상수, 변수 및 연산을 그래프로 조합합니다.
# 세션 내에서 이러한 상수, 변수 및 연산을 평가합니다.

## 간단한 TensorFlow 프로그램 만들기
## import 구문 제공
# 거의 모든 Python 프로그램에서와 마찬가지로 import 문을 지정하여 시작합니다. 물론 TensorFlow 프로그램을 실행하는 데 필요한 import 세트는 프로그램에서 액세스할 기능에 따라 다릅니다. 최소한 모든 TensorFlow 프로그램에서 import tensorflow 문을 제공해야합니다.
import tensorflow as tf

import matplotlib.pyplot as plt # Dataset 시각화. 데이터 시각화를 위해 사용.
import numpy as np # 저수준 수치해석 파이썬 라이브러리. 데이터를 준비하는 과정에 주로 사용.
import pandas as pd # 고수준 수치해석 파이썬 라이브러리. 데이터를 엑셀처럼 고수준에서 다루는 과정에 주로 사용.

# TensorFlow는 기본 그래프default graph를 제공합니다. 그러나 명시적으로 자체 Graph를 만드는 것이 좋습니다. (일반적으로, Estimator와 같은 상위 수준 API를 사용합니다만, 저수준의 TensorFlow 코드를 작성하는 경우 명시 적으로 상태를 추적하도록 그래프를 선언하는 것이 좋습니다. (각 셀에서 다른 그래프로 작업하고 싶을 수도 있으므로, 그래프를 테스트할 때는필수적입니다.)

# graph를 만듭니다.
g = tf.Graph()

# 그래프를 기본 "default"그래프로 지정합니다.
with g.as_default():
  # 아래의 세가지 작업으로 구성된 그래프를 조합합니다:
  #   * 피연산자를 만들기 위한 두개의 tf.constant 작업.
  #   * 두 피연산자를 더하기 위한 하나의 tf.add 작업.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  sum = tf.add(x, y, name="x_y_sum")

  # 이제 세션을 생성합니다.
  # 새션을 기본 그래프를 실행할 것입니다.
  with tf.Session() as sess:
    print(sum.eval())

## 연습 : 코드 블록에 세 번째 피연산자 가져 오기
# 보다 구체적으로 다음을 수행하십시오.

 # 앞의 코드 블록에 세 번째 스칼라 정수 피연산자를 도입하십시오.
 # 세 번째 피연산자를 sum에 추가하면 새로운 합계가 산출됩니다.
 # 수정 된 코드 블록을 다시 실행하십시오. (다른 코드 블록을 재실행 할 필요가 없습니다.) 프로그램이 올바른 총계를 생성 했습니까?

### 텐서 다루기
## 학습 목표

# TensorFlow 변수 초기화 및 할당을 해 봅니다.
# 텐서 생성 및 조작을 익힙니다.
# 선형 대수학에서의 덧셈과 곱셈에 대한 기억을 추억에서 끄집어 냅니다.(이 주제가 처음이라면 행렬의 덧셈과 곱셈을 참고하십시오)
# 기본 TensorFlow 수학 및 배열 작업을 익힙니다.

import tensorflow as tf

## 벡터 추가
# 텐서에서 많은 일반적인 수학 연산을 수행 할 수 있습니다 (TF API. 다음 코드는 정확히 6 개의 요소를 갖는 벡터 (1-D 텐서)를 생성하고 조작합니다
# TF API - https://www.tensorflow.org/api_docs/

with tf.Graph().as_default():
  # 6 요소 벡터 (1-D 텐서)를 만듭니다.
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # 다른 6 요소 벡터를 만듭니다. 벡터의 각 요소는 1로 초기화됩니다. 
  # 첫 번째 인수는 텐서의 모양(shape)입니다 (아래 모양 참조).

  ones = tf.ones([6], dtype=tf.int32)

  # 두 개의 텐서를 더합니다. 결과 텐서는 6 요소 벡터입니다.
  just_beyond_primes = tf.add(primes, ones)

  # 기본 그래프를 실행하는 세션을 만듭니다.
  with tf.Session() as sess:
    print(just_beyond_primes.eval())

## 텐서 모양
# 모양(shape)은 텐서의 크기와 차원 수를 특성화하는 데 사용됩니다. 텐서의 모양은 list로 표현되며 i 번째 요소는 차원 i의 크기를 나타냅니다. 리스트의 길이는 텐서의 랭크 (즉, 차원의 수)를 의미합니다.
# TensorFlow 설명서 - https://www.tensorflow.org/programmers_guide/dims_types

with tf.Graph().as_default():
  # scalar 는 0차원 텐서입니다.
  scalar = tf.zeros([])

  # 3개의 요소를 갖는 벡터.
  vector = tf.zeros([3])

  # 2개의 행과 3개의 열로 구성된 행렬
  matrix = tf.zeros([2, 3])

  with tf.Session() as sess:
    print('scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval())
    print('vector has shape', vector.get_shape(), 'and value:\n', vector.eval())
    print('matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval())

## 브로드캐스팅

# 수학에서는 동일한 모양의 텐서에만 요소 별 연산 (예 : addition 및 equals)을 수행 할 수 있습니다. 그러나 TensorFlow에서는 전통적으로 호환되지 않는 텐서 (tensors)에 대한 연산을 수행 할 수 있습니다. 브로드캐스팅 - numpy에서 빌린 개념 - 은 TensorFlow가 작업 내의 작은 배열을 자동으로 확대하여 요소 단위 작업에 적합한 모양으로 만듭니다.

# 텐서가 브로드캐스트되면, 그 차원을 따르는 항목들은 개념적으로 복사 됩니다. (성능상의 이유로 실제로 복사되지는 않습니다. 브로드캐스팅은 성능 최적화를 하도록 고안되었습니다.)

# 예를 들어, 브로드캐스팅은 다음을 가능하게합니다.

# 피연산자에 크기 [6] 텐서가 필요한 경우 크기 [1] 또는 크기 [] 텐서가 피연산자로 사용될 수 있습니다.
# 연산에 [4, 6] 크기의 텐서가 필요한 경우 다음 크기 중 하나를 피연산자로 사용할 수 있습니다.
 # [1, 6]
 # [6]
 # []
# 연산에 크기 [3, 5, 6] 텐서가 필요한 경우 다음 크기 중 하나를 피연산자로 사용할 수 있습니다.
 # [1, 5, 6]
 # [3, 1 ,6]
 # [3, 5, 1]
 # [1, 1, 1]
 # [5, 6]
 # [1, 6]
 # [6]
 # [1]
 # []

# numpy 브로드캐스팅 문서 - https://docs.scipy.org/doc/numpy-1.10.1/user/basics.broadcasting.html

with tf.Graph().as_default():
  # 6개의 요소로 구성된 벡터 (1차원 텐서)를 만듭니다.
  primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

  # 값이 1인 상수 스칼라를 만듭니다.
  ones = tf.constant(1, dtype=tf.int32)

  # 두 텐서를 더한다. 결과 텐서는 6개의 요소로 구성된 벡터입니다.
  just_beyond_primes = tf.add(primes, ones)

  with tf.Session() as sess:
    print(just_beyond_primes.eval())

## 행렬 곱
# 선형 대수학에서 두 개의 행렬을 곱할 때 첫 번째 행렬의 열 수는 두 번째 행렬의 행 수와 같아야합니다.

 # 3x4 행렬에 4x2 행렬을 곱하는 것은 유효합니다. 이렇게하면 3x2 행렬이됩니다.
 # 4x2 행렬에 3x4 행렬을 곱하는 것은 유효하지 않습니다.

with tf.Graph().as_default():
  # 3 행 4 열의 행렬 (2-D 텐서)을 만듭니다.
  x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                  dtype=tf.int32)

  # 4 행 2 열의 행렬을 만듭니다.
  y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

  # 두 피연산자 행렬을 곱합니다.
  # 결과 행렬은 3 행 2 열이 될 것입니다.
  matrix_multiply_result = tf.matmul(x, y)

  with tf.Session() as sess:
    print(matrix_multiply_result.eval())

## 텐서 모양 변형
# 텐서 합과 행렬 곱셈이 피연산자에 큰 제약을 부과하는 경우 TensorFlow 프로그래머는 종종 텐서를 변형해야합니다. tf.reshape 메서드를 사용하여 텐서를 다시 만듭니다. 예를 들어, 20x2 텐서를 2x20 텐서 또는 10x4 텐서로 바꿀 수 있습니다. tf.reshape를 사용하여 텐서의 차원 수 ("랭크")를 변경할 수도 있습니다. 예를 들어, 20x2 텐서를 1-D 40 요소 텐서 또는 3-D 5x2x4 텐서로 바꿀 수 있습니다.

with tf.Graph().as_default():
  # 4개의 요소 벡터를 만듭니다.
  a = tf.constant([5, 2, 4, 3], dtype=tf.int32)

  # 4 요소 벡터를 2x2 행렬로 모양을 변형합니다.
  reshaped_matrix = tf.reshape(a, [2,2])

  with tf.Session() as sess:
    print(reshaped_matrix.eval())

## 연습 1: 두 개의 텐서를 곱하기 위해 변형 시키십시오.
# 다음 두 벡터는 행렬 곱셈과 호환되지 않습니다.

 # a = tf.constant ([5, 3, 2, 7, 1, 4])
 # b = tf.constant ([4, 6, 3])
# 이 벡터를 행렬 곱셈과 호환되는 피연산자로 변형해 봅시다. 그런 다음, 변형한 텐서끼리 행렬 곱셈 연산을 수행해봅시다.

# 위 두 벡터를 이용해서 (3, 6) 모양을 가진 행렬을 만들어보세요. (단, 행렬 곱셈 tf.matmul 을 이용하시오)

import tensorflow as tf
a = tf.constant ([5, 3, 2, 7, 1, 4])
b = tf.constant ([4, 6, 3])
#
# 여기에 코드를 입력하세요.
#

## 변수, 초기화 및 할당
# 지금까지 우리가 보여준 모든 개념은 상태 비의존적이었습니다. eval()을 호출하면 항상 동일한 결과가 반환됩니다. TensorFlow 변수 개체(tf.Variable)를 사용하면 값을 바꾸는 것도 가능합니다. 또한, 상수(Constant)를 초기값으로 줄 수도 있고, 정규분포를 초기화 함수(initializer)로 사용하여 초기화하는 것도 가능합니다.

g = tf.Graph()
with g.as_default():
  # 초기값이 3인 변수를 만든다
  v = tf.Variable([3])

  # 형태가 [1]인 변수를 랜덤값으로 초기화하여 만든다.
  # 랜덤은 평균 1, 표준편자가 0.35인 정규 분포로부터 샘플링한다.
  w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))

# TensorFlow의 특징 중 하나는 변수 초기화가 자동으로 수행되지 않는다 는 것입니다. 예를 들어 다음 블록은 오류를 발생시킵니다.

with g.as_default():
  with tf.Session() as sess:
    try:
      v.eval()
    except tf.errors.FailedPreconditionError as e:
      print("Caught expected error: ", e)

# 변수를 초기화하는 가장 쉬운 방법은 global_variables_initializer를 호출하는 것입니다. Session.run() 사용은 eval() 과 거의 같습니다.

with g.as_default():
  with tf.Session() as sess:
    initialization = tf.global_variables_initializer()
    sess.run(initialization)
    # 이제, 변수는 평소처럼 읽을 수 있습니다. 할당된 값들이 들어 있습니다다.
    print(v.eval())
    print(w.eval())

# 변수는 초기화되면 동일한 세션 내에서 값을 유지합니다. 그러나 새 세션을 시작할 때 세션을 다시 초기화해야합니다.
with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 아래의 세 출력은 같은 값을 출력할 것입니다.
    print(w.eval())
    print(w.eval())
    print(w.eval())

# 변수의 값을 변경하려면 할당assign op를 사용하십시오. assign op를 만드는 것 만으로는 아무 효과가 없습니다. 초기화와 마찬가지로 원하는 효과를 얻기 전에 할당 op를 실행해야합니다.
# 변수의 값을 변경하기 위해서는 tf.assign(variable, value) 을 사용하십시오. assign op를 만드는 것 만으로는 동작하지 않습니다. 우선 정의한 후, assign 을 실행해야 실제 반영이 됩니다.

# 주의할 점은

 # tf.assign 을 사용해 operation을 만들고
 # 만들어진 operation을 session을 통해 실행해야 비로소 값이 바뀐다는 점입니다.

 with g.as_default():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 이 명령은 변수의 초기값을 출력합니다.
    print(v.eval())

    assignment = tf.assign(v, [7])
    # 변수는 아직 변경되지 않았음!
    print(v.eval())

    # assignment op. 를 실행합니다.
    sess.run(assignment)
    # 이제 변수가 업데이트 되었습니다.
    print(v.eval())

# TensorFlow 문서 - https://www.tensorflow.org/programmers_guide/variables

## 연습 2 : 두 주사위를 10회 던지는 경우를 시뮬레이션 하십시오.

# 주사위 시뮬레이션을 만들어 (10, 3) 모양의 2-D 텐서를 완성하십시오.

 # 열 1과 2는 각각 하나의 주사위를 던진 값입니다.
 # 3 열은 동일한 행에 있는 두 개의 던진 값의 합계를 유지합니다.

# 예를 들어 첫 번째 행은 다음과 같은 열 값을 포함 할 수 있습니다.

 # 열 1은 4를 저장합니다.
 # 열 2는 3을 저장합니다.
 # 열 3은 7을 저장합니다.

### TensorFlow 시작하기
# 학습 목표

 # 기본적인 TensorFlow 개념을 학습합니다.
 # TensorFlow의 LinearRegressor 클래스를 사용하여 하나의 입력 특징을 기반으로 도시 블록의 입도에 따른 주택 가격의 중간 값을 예측합니다.
 # RMSE (Root Mean Squared Error)를 사용하여 모델의 예측 정확도 평가
 # 하이퍼 매개변수 (hyper parameters)를 조정하여 모델의 정확성을 향상시킵니다.

# 캘리포니아의 1990년 인구 조사 데이터를 사용합니다.

## 설정
# 파이썬 기본 수학 라이브러리
import math

# 시각화 라이브러리들
# from sorna.display import display
from IPython.display import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

# 수치 해석 및 데이터 핸들링 라이브러리들
import numpy as np
import pandas as pd

# 기계학습 라이브러리들
# learn_io 는 pd.DataFrame 을 TensorFlow로 불러오기 위해 사용됩니다. 
# estimator 는 TensorFlow 모델 작성을 위한 고수준 API입니다.
# 자세한 설명은 아래에서 다시 보게 될 것입니다.

from sklearn import metrics # sklearn 설치 오류
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io, estimator

# 기타 로그 옵션 (중요하지 않습니다)
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

 # learn_io 는 pd.DataFrame 을 TensorFlow로 불러오기 위해 사용됩니다.
 # estimator 는 TensorFlow 모델 작성을 위한 고수준 API입니다.
# 그런 다음 데이터셋을 불러옵니다.

california_housing_dataframe = pd.read_csv("http://datasets.lablup.ai/public/tutorials/california_housing_train.csv", sep=",")

# 확률경사하강법(Stochastic Gradient Descent)의 성능에 해를 끼칠 수 있는 순서에 의한 영향을 받지 않도록 데이터를 무작위로 추출합니다. 또한 median_house_value를 천단위로 조정하여, 일반적으로 사용하는 범위의 학습율로 조금 더 쉽게 배울 수 있습니다.

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
display(california_housing_dataframe.head(100))

## 데이터 검사
# 작업하기 전에 데이터를 조금이라도 알고있는 것이 좋습니다.

# 각 열에 대한 유용한 통계에 대한 간단한 요약을 확인할 것입니다. 여기에는 평균, 표준 편차, 최대, 최소 및 다양한 quantile과 같은 항목이 포함됩니다.

display(california_housing_dataframe.describe())

## 경사하강법
# 경사 하강법은 함수의 기울기(그라디언트)를 이용하여 손실을 최소화 하는 최적화 방법입니다. 뒤에서 GradientDescentOptimizer를 이용하여 사용해 보겠습니다.

# 경사 하강법을 실행하는 모습. x0x0에서 시작하여, 경사가 낮아지는 쪽으로 이동하여 차례대로 x값을 얻습니다.

# 첫 번째 모델 만들기
# 이 연습에서는 median_house_value를 예측하려고합니다. 이 값이 우리의 레이블 (때로는 대상이라고도 함)이 됩니다. total_rooms를 입력 특징으로 사용하겠습니다.

# 이 데이터는 도시 블록 수준에 해당하므로, 이 특징은 해당 블록의 총 객실 수 또는 해당 블록에 거주하는 총 사용자 수를 각각 반영합니다.

# 모델을 학습하기 위해 TensorFlow contrib.learn 라이브러리에서 제공하는 LinearRegressor 인터페이스를 사용합니다. 이 라이브러리는 많은 입출력 파이프라인과 사용 가능하며, 데이터, 훈련 및 평가 과정을 편리하게 할 수 있는 인터페이스를 제공합니다.

# 먼저 입력 특징(feature) 및, 대상을 정의하고 LinearRegressor 객체를 만듭니다.

# GradientDescentOptimizer는 미니배치 확률경사하강법 (Mini-Batch Stochastic Gradient Descent,SGD)를 구현합니다. 여기에서 mini-batch의 크기는 batch_size 매개 변수로 지정됩니다. optimizer의 learning_rate 매개 변수에 유의하십시오.이 매개 변수는 그라디언트 단계의 크기를 제어합니다. 안전을 위해 gradient_clip_norm 값도 포함합니다. 이것은 그라디언트가 너무 커서 그라데이션 강하에서 나쁜 결과가 나오는 경우를 피하도록 도와줍니다.

my_feature = california_housing_dataframe[["total_rooms"]]
targets = california_housing_dataframe["median_house_value"]

training_input_fn = learn_io.pandas_input_fn(
    x=my_feature, y=targets, num_epochs=None, batch_size=1)

feature_columns = [tf.contrib.layers.real_valued_column("total_rooms", dimension=1)]

linear_regressor = tf.contrib.learn.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00001),
    gradient_clip_norm=5.0,
)

# 특징 열과 대상 (레이블)에 fit()을 호출하면 모델을 학습하게 됩니다.

_ = linear_regressor.fit(
    input_fn=training_input_fn,
    steps=100
)

#  훈련 데이터에 대한 예측을 만들어 훈련 데이터에 얼마나 잘 맞는지 봅시다.

prediction_input_fn = learn_io.pandas_input_fn(
    x=my_feature, y=targets, num_epochs=1, shuffle=False)

predictions = list(linear_regressor.predict(input_fn=prediction_input_fn))
mean_squared_error = metrics.mean_squared_error(predictions, targets)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % math.sqrt(mean_squared_error))

## 모델 평가
# 좋아요, 모델 훈련은 쉬웠습니다! 그러나 이것은 좋은 모델일까요? 오류의 크기를 어떻게 판단 하시겠습니까?

# 평균 제곱 오류는 해석하기가 어려울 수 있으므로 대신 RMSE (평균 제곱근 편차)를 참조하십시오. RMSE는 원래 대상과 동일한 축척으로 해석 될 수있는 좋은 특성을 가지고 있습니다.

# RMSE를 실제값(y)의 최소값~최대값 범위와 비교하여 대략적으로 RMSE 값이 얼마나 큰 값인지 알 수 있습니다.

# 더 잘 할 수 있을까요?

# 이것은 모든 모델 개발자에게 중요한 질문입니다. 몇 가지 지침을 제공하는 데 도움이되는 몇 가지 기본 전략을 개발해 보겠습니다.

# 우리가 할 수 있는 첫 번째 일은 전체 요약 통계의 관점에서 우리의 예측이 우리의 목표와 얼마나 잘 일치하는지 살펴 보는 것입니다.

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
display(calibration_data.describe())

# 이 정보가 도움이 될지도 모릅니다. 평균값은 모델의 RMSE와 어떤 차이가 있습니까? 다양한 분위수(quantile)는 어떻습니까?

# 우리는 또한 우리가 배운 데이터와 선을 시각화 할 수 있습니다. 단일 특징에 대한 선형 회귀는 입력을 출력 x로 매핑하는 선으로 그릴 수 있습니다.

# 먼저 데이터의 일정한 무작위 표본을 얻습니다. 이것은 산포도(scatter plot)을 읽기 쉽게 만드는 데 유용합니다.

sample = california_housing_dataframe.sample(n=300)

# 그런 다음 모형의 편향(bias) 항 및 특징 가중치를 함께 스캐터 플롯으로 그립니다. 선이 빨간색으로 보일 것입니다.

x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()
y_0 = linear_regressor.weights_[0] * x_0 + linear_regressor.bias_
y_1 = linear_regressor.weights_[0] * x_1 + linear_regressor.bias_
plt.plot([x_0, x_1], [y_0, y_1], c='r')
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")
plt.scatter(sample["total_rooms"], sample["median_house_value"])
plt.show()

# 예측 모델(빨간선)을 보면 데이터를 제대로 예측하지 못하고 있 음을 알 수 있습니다. 위 요약 통계 (calibration_data.describe())를 다시 보고 이러한 사실을 알아 낼 수 있는지 확인하십시오. 이러한 초기 온전성 검사 과정을 통해 우리는 좀더 나은 모델을 찾아야 함을 알 수 있습니다.

## 모델 매개 변수 조정
# 이 연습에서는 위의 모든 코드를 편의를 위해 단일 함수에 넣었습니다. 다른 매개 변수로 함수를 호출하여 그 영향을 볼 수 있습니다.

# 이 함수에서 우리는 각 기간에 모델 개선을 관찰 할 수 있도록 균등하게 분할 된 10 개의 기간으로 진행할 것입니다.

# 각 기간에 대해 훈련 손실을 계산하고 이를 그래프로 나타냅니다. 이는 모델이 수렴되는 시기를 판단하거나 반복이 더 필요한지 판단하는 데 도움이 될 수 있습니다.

# 시간이 지남에 따라 모델의 가중치(weights)와 편향(bias)값의 변화 과정을 시각화하여 모델이 수렴했는지 확인할 수 있습니다.

def train_model(learning_rate, steps, batch_size, input_feature="total_rooms"):
  """하나의 특징에 대한 선형 회귀 모델을 훈련합니다.

  Args:
    learning_rate: `float`, 학습율.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 훈련 단계는 단일 배치를 사용하며,
      forward 및 backward 패스로 구성됩니다.
    batch_size: 0이 아닌 `int`, 배치 크기.
    input_feature: 입력 특징으로 사용하기 위한 `california_housing_dataframe`의 지정한 열.  `string`
  """

  periods = 10
  steps_per_period = steps / periods

  # total_rooms 의 데이터를 이용하여 median_house_value 를 예측하는 모델을 만들겠습니다.
  # 이를 위해 다음과 같이 파이프라인을 정의합니다.
  my_feature = input_feature
  my_feature_column = california_housing_dataframe[[my_feature]]
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label]

  # 특징 열 만들기
  feature_columns = [tf.contrib.layers.real_valued_column(my_feature, dimension=1)]

  # 입력 함수들 만들기
  # 특징 열과 대상 (레이블)에 fit()을 호출하면 모델을 학습하게 됩니다. 
  # 학습 데이터는 learn_io.pandas_input_fn 을 이용하여 미니배치 형태로 구성됩니다.
  training_input_fn = learn_io.pandas_input_fn(
    x=my_feature_column, y=targets, num_epochs=None, batch_size=batch_size)
  prediction_input_fn = learn_io.pandas_input_fn(
    x=my_feature_column, y=targets, num_epochs=1, shuffle=False)

  # 선형 회귀 오브젝트 만들기
  linear_regressor = tf.contrib.learn.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0
  )

  # 각 주기별로 모델의 상태를 플롯하기 위해 준비
  plt.clf()
  plt.close()
  plt.figure(figsize=(15, 6))
  plt.subplot(1, 2, 1)
  plt.title("Learned Line by Period")
  plt.ylabel(my_label)
  plt.xlabel(my_feature)
  sample = california_housing_dataframe.sample(n=300)
  plt.scatter(sample[my_feature], sample[my_label])
  colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]

  # 모델을 훈련 시키되 루프 내부에서 수행하여 손실 매트릭을 주기적으로 평가할 수 있습니다.
  print("Training model...")
  print("RMSE (on training data):")
  root_mean_squared_errors = []
  for period in range (0, periods):
    # 이전 상태에서 시작하여 모델을 교육.
    linear_regressor.fit(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # 잠시 멈추고 예측을 계산합니다.
    predictions = list(linear_regressor.predict(
        input_fn=prediction_input_fn))
    # 손실 계산.
    root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(predictions, targets))
    # 주기적으로 현재의 손실을 출력.
    print("  period %02d : %0.2f" % (period, root_mean_squared_error))
    # 이번 주기의 손실 매트릭을 리스트에 추가.
    root_mean_squared_errors.append(root_mean_squared_error)
    # 마지막으로 시간 경과에 따라 가중치와 편향을 추적합니다.
    # 몇 가지 수학을 적용하여 데이터와 선이 깔끔하게 정리되도록 합니다.
    y_extents = np.array([0, sample[my_label].max()])
    x_extents = (y_extents - linear_regressor.bias_) / linear_regressor.weights_[0]
    x_extents = np.maximum(np.minimum(x_extents,
                                      sample[my_feature].max()),
                           sample[my_feature].min())
    y_extents = linear_regressor.weights_[0] * x_extents + linear_regressor.bias_
    plt.plot(x_extents, y_extents, color=colors[period]) 
  print("Model training finished.")

  # 주기에 따른 손실 매트릭 그래프 출력
  plt.subplot(1, 2, 2)
  plt.ylabel('RMSE')
  plt.xlabel('Periods')
  plt.title("Root Mean Squared Error vs. Periods")
  plt.plot(root_mean_squared_errors)
  plt.show()
  # 보정 데이터가 있는 표를 출력합니다.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

## 과제 1 : 손실을 개선하고 대상 분포에 더 맞게 조정하기
# 목표는 RMSE에서 약 180 점 이하의 값을 얻으려고 시도하는 것입니다.

# 시도한 지 5 분이 지나도 180점 이하의 RMSE를 얻지 못했을 때 가능한 조합에 대한 답안을 확인하십시오.

train_model(
    learning_rate=0.00001,
    steps=100,
    batch_size=1
)

## 모델 튜닝이란?

# 모델의 가중치(weights) 와 편향값(bias)을 매개변수라고 하고 이 외에 학습을 시작하기 전에 정해지는 값들을 하이퍼 매개변수(hyper parameters)라고 합니다.

# 다음과 같은 값들입니다.

 # 학습 속도(learning rate)
 # 배치 사이즈

# 최적의 하이퍼 매개변수를 찾는 과정을 모델 튜닝 이라고 합니다.

## 모델 튜닝을 위한 표준 방법이 있습니까?

# 자주 묻는 질문입니다. 간단히 말해서 하이퍼 매개 변수들의 영향은 데이터에 따라 다릅니다. 그래서 쉽고 빠른 규칙이 없습니다. 데이터에 대해 실험을 반복하는 것만이 유일한 방법입니다.

# 다음은 모델 튜닝을 돕기 위한 팁입니다.

 # 훈련 오류(error)는 꾸준히 감소하고 처음에는 가파르게 증가하다 일정한 값으로 수렴되면서 평평한 상태에 놓일 것입니다.
 # 모델이 수렴되지 않은 경우 더 오래 실행하십시오.
 # 학습 오류가 너무 느리게 감소하면 학습율 (learning rate)을 높이면 학습 속도가 빨라집니다.
  # 그러나 학습 속도가 너무 높으면 오히려 학습이 불가능해질 수 있습니다.
 # 훈련 오류가 격렬하게 변하는 경우 학습율을 줄이십시오.
  # 학습률이 낮을 경우 더 많은 단계와 더 큰 배치 크기로 결과를 좋게 만들 수 있습니다.
 # 배치 크기가 매우 작으면 불안정 할 수도 있습니다. (일반적으로 64, 128, 256, 512, 1024 등의 값을 선택합니다.) 먼저 100이나 1000과 같은 더 큰 값을 시도해보세요. 모델의 크기와 메모리 사이즈에 영향을 받기 때문에 메모리가 부족하면 배치 크기를 줄이십시오.

# 위 방법은 데이터에 따라 다르기 때문에 이러한 규칙을 과신하지 마십시오. 항상 실험하고 확인하십시오

## 과제 2 : 다른 특징을 사용해 보기

# total_rooms 특징을 population 특징으로 대체하여 더 잘 수행 할 수 있는지 확인하십시오.