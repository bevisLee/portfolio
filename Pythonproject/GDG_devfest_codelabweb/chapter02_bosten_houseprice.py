
## url - https://codeonweb.com/entry/f7b88b3b-29b2-4b15-b48c-2149b81bf409

import keras
print(keras.__version__)

"""
집값 예측하기 : 회귀분석 예제 (regression)

일반적인 기계 학습 문제는 연속적인 값을 학습해서 예측하는 "회귀(regression)" 입니다. 
예를 들어, 기상 데이터를 가지고 내일 기온을 예측하거나, 명세서를 가지고 소프트웨어 개발 프로젝트 소요 시간을 예측하거나 등의 문제입니다.

"회귀분석"를 "로지스틱 회귀(logistic regression)"와는 다릅니다. 
역사적인 이유로 회기(regression)이라고 붙었지만, "로지스틱 회귀"는 회귀 알고리즘이 아니고 분류 알고리즘입니다;;
보스턴 집값 데이터셋

1970년대 중반 보스턴 교외 지역의 범죄율, 지방 재산세율 등등의 당시 데이터를 가지고 주택의 중앙값(midle value)을 예측하겠습니다.

데이터셋은 전체 데이터가 506개밖에 안 되어서 학습 데이터 404개, 평가 데이터 102개로 나누어 사용합니다.
학습 데이터 404개
평가 데이터 102개
입력 데이터의 각 "특징"(범죄율 같은)이 서로 다른 스케일을 가지고 있습니다.
예를 들어 어떤 특징은 0에서 1 사이 값을 가지는 비율이며,
어떤 특징은 1에서 12 사이의 값을 가지고,
또 다른 값은 0에서 100 사이의 값을 가지는 등등...
그럼 데이터를 한번 살펴볼까요:
"""

from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

print(train_data.shape) # 학습 데이터 404개, 13개 특징
print(test_data.shape) # 평가 데이터 102개, 13개 특징

"""
데이터 셋의 특징

404개의 훈련 샘플과 102개의 시험 샘플이 있습니다. 
데이터는 13개의 특징(feature)을 가지고 있습니다. 
입력 데이터의 13가지 특징은 다음과 같습니다:

1인당 범죄율.
25,000 평방 피트 이상으로 구획된 주거 용지의 비율.
동네 별 비상업 용지 비율.
찰스 강 가변수(강가에 접해 있으면 1, 아니면 0).
질소 산화물 농도(천만분의 일 단위).
주택 당 평균 방 갯수.
1940년 이전에 지어진 자가 주택 비율.
5개 보스턴 고용 센터까지의 가중 거리.
방사형 고속도로에의 접근성 지수.
10,000 달러 당 전체가치재산세율.
동네 별 학생-교사 비.
Bk를 동네의 흑인 비율이라 할 때 1000 * (Bk - 0.63) ** 2
인구 중 낮은 사회적 지위의 백분율.
목표: 보스턴 집값의 통계적인 중앙값 구하기

목표는 자가 주택 가격의 중앙값이며 단위는 1,000 달러 입니다:
"""

print(train_targets)

"""
가격은 보통 10,000 달러에서 50,000 달러 사이이군요. 너무 싸다 싶으면, 이게 1970년대의 가격이고 인플레이션이 보정되지 않았다는 점을 감안하세요.

데이터 준비하기

이렇게 범위가 완전히 다른 값을 신경망에 바로 넣는 건 문제가 있죠. 신경망이 이렇게 형태가 다른 데이터에 자동으로 적응을 하긴 하겠지만, 학습하기에는 분명히 훨씬 더 어려울 겁니다.

표준화 (normalization) 하기
이런 데이터를 다루는 데에 가장 널리 쓰이는 방법은 특징 별로 표준화(normalization)하는 겁니다.
입력 데이터의 각 특징(입력 데이터 행렬의 열)에 대해서 평균을 빼고 표준 편차로 나누어서 각 특징이 0 근처를 중심으로 하고 단위 표준 편차를 가지게 합니다. 
이 작업은 Numpy 로 간단하게 할 수 있습니다:
"""

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

"""
시험 데이터에 적용할 표준화 값 역시 훈련 데이터를 사용하여 구한다는 것에 유의하세요. 작업 흐름 상에서 시험 데이터에 대해서 계산한 값은 그 무엇도 사용하여서는 안 됩니다, 데이터 표준화에 관련된 이런 사소한 것까지도요.

신경망 쌓기

상당히 적은 수의 샘플 밖에 없기 때문에 각각 64 단위를 가지는 2 개의 은닉 계층으로 된 아주 작은 신경망을 사용하도록 하겠습니다. 일반적으로 훈련 데이터가 적을 수록 과적합이 더 심하기 때문에 작은 신경망을 사용하는 것이 과적합을 완화하는 한 방법입니다.
"""

from keras import models
from keras import layers

def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

"""
신경망은 활성화 없이 단일 단위로 끝납니다(즉, 선형 계층입니다). 스칼라 회귀(단일 연속 값을 예측하는 회귀)에서는 전형적인 설정입니다. 활성화 함수를 적용하면 출력값의 범위가 제한될 수 있습니다; 예를 들어 마지막 계층에 sigmoid 활성화 함수를 적용하면 신경망은 0과 1 사이의 예측 값만을 학습할 수 있습니다. 여기서는 마지막 계층이 단순히 선형이기 때문에 신경망은 어더한 범위의 예측 값도 학습할 수 있습니다.

신경망을 mse 손실 함수를 써서 컴파일할 거라는 점에 주목하세요. 평균 제곱 오차(Mean Squared Error)는 예측과 목표 간 차이의 제곱으로 회귀 문제의 손실 함수로 널리 쓰입니다.

그리고 훈련 중에 새로운 메트릭, mae를 모니터링할 겁니다. 평균 절대 오차(Mean Absolute Error)는 예측과 목표 간 차이의 절대값입니다. 예를 들어 이 문제에서 평균 절대 오차가 0.5라면 예측이 평균 500 달러 정도 차이가 있다는 뜻입니다.

K-fold 검증으로 접근법 검증하기

파라미터(훈련 epoch 횟수와 같은)를 계속 조정해 가면서 신경망을 평가하려면, 앞선 예제에서와 같이 그냥 데이터를 훈련 세트와 검증 세트로 나누면 됩니다. 하지만 지금은 데이터가 얼마 없기 때문에 검증 세트 역시 매우 적을 수 있습니다(100개 정도). 이렇게 되면 결과적으로 검증 점수가 어떤 데이터를 검증 데이터로 고르고, 훈련 데이터로 고르는지에 의존해 버리게 됩니다. 즉, 검증 데이터를 어떻게 나누느냐에 따라 검증 점수의 분산이 너무 높아지게 됩니다. 이러면 모델을 신뢰성 있게 평가하기 힘듭니다.

이런 상황에서 가장 좋은 방법은 K-fold 교차 검증입니다. 사용 가능한 데이터를 K개의 부분으로 나누고(K는 보통 4나 5) K개의 동일한 모델을 인스턴스화 하여서 각 모델을 K-1 부분의 데이터로 훈련하고 나머지 한 부분의 데이터로 평가합니다. 모델의 검증 점수는 얻어진 K개의 검증 점수의 평균이 됩니다.

코드로 보면 바로 이해가 됩니다:
"""

import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)

print(np.mean(all_scores))

"""
확인할 수 있는 바와 같이 각각의 시행에서 실제로 2.1 에서 2.9까지 상당히 다른 값이 보여집니다. 그 평균(2.4)은 다른 어떤 단일 점수보다 훨씬 더 신뢰할 만한 메트릭이죠. 바로 이 것이 K-fold 교차 검증의 핵심입니다. 이 경우에는 평균에서 2,400 달러씩 떨어져 있는데 가격이 10,000 달러에서 50,000 달러 사이라는 점을 감안하면 아직 꽤 큰 오차입니다.

그럼 신경망을 좀 더 훈련시켜 봅시다: 500 epoch 동안. 각 epoch 마다 모델이 잘 하고 있는지를 기록하기 위해 epoch마다 검증 점수 로그를 저장하도록 훈련 과정을 수정하도록 하겠습니다.
"""

from keras import backend as K

# Some memory clean-up
K.clear_session()

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # Build the Keras model (already compiled)
    model = build_model()
    # Train the model (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

"""
이제 모든 fold에 대해 epoch 당 평균 절대 오차 점수의 평균을 계산할 수 있습니다:
"""

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

"""
이걸 그래프로 그려 봅시다:
"""

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
그래프로 보기가 스케일링 문제와 상대적으로 높은 분산 때문에 조금 어려울 수도 있습니다. 그럼 이렇게 해봅시다:

커브의 나머지 부분과 스케일이 다른 제일 앞의 10개 데이터를 생략합니다.
매끄러운 커브를 얻기 위해 각 점을 앞 점의 지수 이동 평균(exponential moving average)으로 대체합니다.
"""

def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.clf()
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
이 그래프에 의하면, 80 epoch 이후에는 검증 평균 절대 오차가 유의미하게 개선되는 것을 멈춥니다. 이 지점을 지나면 과적합이 시작되는 거죠.

모델의 다른 파라미터까지 조정하고 나면(epoch 횟수 외에도 은닉 계층의 크기도 조정할 수 있습니다), 이제 최종 "제품" 모델을 최적의 파라미터로 모든 훈련 데이터에 대해 훈련할 수 있습니다. 그럼, 시험 데이터에 대해서 성능을 한번 봅시다:
"""

# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)

"""
아직도 2,550 달러 정도 떨어져 있네요.

마무리

이 예제의 시사점은 다음과 같습니다:

회귀는 사용하는 손실 함수가 분류와는 다릅니다; 평균 제곱 오차(MSE, Mean Squared Error)는 회귀에 일반적으로 쓰이는 손실 함수 입니다.
비슷하게, 회귀에 사용하는 평가 메트릭 역시 분류와는 다릅니다; 당연하게도 "정확도" 개념은 회귀에는 적용되지 않습니다. 일반적인 회귀 메트릭은 평균 절대 오차(MAE, Mean Absolute Error)입니다.
입력 데이터의 특징이 서로 다른 범위를 가질 때, 전처리 단계에서 각 특징은 독립적으로 스케일링이 되어야 합니다.
사용할 수 있는 데이터가 적을 때, K-Fold 검증은 모델을 평가할 때에 신뢰할 수 있는 좋은 방법입니다.
사용할 수 있는 데이터가 적을 때, 지나친 과적합을 방지하기 위해 적은 은닉 계층(보통 한개나 두개)을 사용한 작은 신경망 사용이 선호됩니다.
이 예제는 입문용 3 개의 실습 예제중 하나입니다.
입문용 3가지 예제를 통하여 이제 여러분은 다음과 같은 벡터 입력 데이터를 가진 일반적인 종류의 문제를 다룰 수 있습니다:

이진(2개의 범주) 분류.
다중 범주, 단일 레이블 분류.
스칼라 회귀
"""
