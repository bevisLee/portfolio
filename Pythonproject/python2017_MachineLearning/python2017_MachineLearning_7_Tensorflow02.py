### 합성 특징과 특이값

## 학습 목표

 # 다른 두 특징의 비율인 합성 특징을 만들어 봅니다.
 # 새로운 특징을 선형 회귀 모델의 입력으로 사용해 봅니다.
 # 입력 데이터에서 특이값을 식별하고 클리핑 (제거)하여 모델의 효율성을 향상시킵니다.

# 아래 셀의 코드는 이전 강의의 코드와 동일합니다.

import math

# from sorna.display import display
from IPython.display import display

from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scikit_learn import datasets, metrics

import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("http://datasets.lablup.ai/public/tutorials/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
display(california_housing_dataframe.head(100))

def train_model(learning_rate, steps, batch_size, input_feature):
  """선형 회귀 모델을 훈련합니다.

  Args:
    learning_rate: `float`, 학습율.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    batch_size: 0이 아닌 `int`, 배치 크기.
    input_feature: 입력 특징으로 쓰기 위하여 `california_housing_dataframe`에서 지정한
      열 이름 `string`.

  Returns:
    모델 훈련 후 목표 및 그에 해당하는 예측을 담은 Pandas `DataFrame`.
  """

  periods = 10
  steps_per_period = steps / periods

  my_feature = input_feature
  my_feature_column = california_housing_dataframe[[my_feature]].astype('float32')
  my_label = "median_house_value"
  targets = california_housing_dataframe[my_label].astype('float32')

  # 입력 함수들 만들기
  training_input_fn = learn_io.pandas_input_fn(
     x=my_feature_column, y=targets,
     num_epochs=None, batch_size=batch_size)
  predict_training_input_fn = learn_io.pandas_input_fn(
     x=my_feature_column, y=targets,
     num_epochs=1, shuffle=False)

  # 선형 회귀 객체 만들기
  feature_columns = [tf.contrib.layers.real_valued_column(my_feature)]
  linear_regressor = tf.contrib.learn.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0
  )

  # 각 주기별로 모델의 상태를 플롯하기 위해 준비
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
        steps=steps_per_period,
    )
    # 잠시 멈추고 예측을 계산합니다.
    predictions = list(linear_regressor.predict(input_fn=predict_training_input_fn))
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
  plt.tight_layout()
  plt.plot(root_mean_squared_errors)

  # 보정 데이터가 있는 표를 출력합니다.
  calibration_data = pd.DataFrame()
  calibration_data["predictions"] = pd.Series(predictions)
  calibration_data["targets"] = pd.Series(targets)
  display(calibration_data.describe())

  print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

  return calibration_data

## 작업 1 : 합성 특징을 시도해봅시다.

# total_rooms 및 population 특징은 주어진 도시 블록에 대한 합계를 계산합니다.

# 그러나 한 도시 블록이 다른 도시 블록보다 인구 밀도가 높으면 어떻게 될까요?

# 우리는 두 가지 특징의 비율인 합성 특징을 만들어 시도할 수 있습니다.

# rooms_per_person이라는 특징을 만들고 이를 아래의 모델 코드 셀에서 입력으로 사용하십시오.

# 이 단일 특징으로 얻을 수있는 최상의 성능은 무엇입니까? 성능이 좋을수록 회귀 직선이 데이터에 적합해야하며 최종 RMSE가 낮아야합니다.

#
# 여기에 코드를 입력하세요.
#
california_housing_dataframe["rooms_per_person"] = 

calibration_data = train_model(
    learning_rate=0.00005,
    steps=500,
    batch_size=5,
    input_feature="rooms_per_person"
)

## 작업 2 : 특이값 확인하기

# 예측 값과 목표 값의 산점도(scatter plot)를 작성하여 모델의 성능을 시각화 할 수 있습니다. 이상적으로, 이들은 완벽하게 상관 관계가 있는 대각선에 위치 할 것입니다.

# 작업 1에서 rooms_per_person 으로 훈련한 모델로 Pyplot의 scatter()를 이용해 산점도를 그려봅시다.

#
# 여기에 코드를 입력하세요.
#
import matplotlib.pyplot as plt

#여기에 코드를 입력하세요
plt.clf()
plt.close()
plt.scatter(, )
plt.show()

# 이상한 점이 보이십니까? rooms_per_person에 있는 값의 분포를 보고 이것들이 어디서 유래했는지 따라가 봅시다.

## 작업 3 : 특이값 제외하기

# rooms_per_person의 특이값 (outlier) 값을 일정한 최소 또는 최대치로 설정하여 모델 적합성을 향상시킬 수 있는지 확인하십시오.

# 참고로, Pandas의 Series기능을 적용하는 방법에 대한 간단한 예제가 있습니다.

clipped_feature = my_feature_column [ "my_feature_name"].apply(lambda x: max(x, 0))

### 검증
## 학습 목표

 # 모델의 효율성을 향상시키기 위해 단일 특징 대신 여러 특징들을 사용해 봅니다.
 # 모델 입력 데이터의 디버그 문제를 짚어 봅니다.
 # 테스트 데이터 세트를 사용하여 모델이 검증 데이터를 오버피팅하는지 확인해 봅니다.

# 이전 실습과 마찬가지로, 1990년 인구 조사 데이터를 대상으로 도시 블록 수준에서 median_house_value를 시도하고 예측하기 위해 캘리포니아 주택 데이터 세트와 함께 작업하겠습니다.

# 먼저 데이터를 불러오고 준비해 봅시다.

import math

# from sorna.display import display
from IPython.display import display

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("http://datasets.lablup.ai/public/tutorials/california_housing_train.csv", sep=",")

def preprocess_features(california_housing_dataframe):
  """캘리포니아 주거 데이터 세트로부터 입력 특징을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    모델에서 사용할 (합성 특징들을 포함한) 특징이 든 DataFrame.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # 합성 특징 만들기.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """캘리포니아 주거 데이터로부터 목표 특징 (레이블)들을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    목표 특징을 포함한 DataFrame.
  """
  output_targets = pd.DataFrame()
  # 목표를 1000달러 단위로 스케일.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

# 훈련 데이터로 17000개 중에서 처음 12000개를 선택하겠습니다.

training_examples = preprocess_features(california_housing_dataframe.head(12000))
display(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
display(training_targets.describe())

# 검증 데이터로 17000개 중에서 마지막 5000개를 선택하겠습니다.

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
display(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
display(validation_targets.describe())

## 데이터 검사

# 위의 데이터를 살펴 보겠습니다. 사용할 수 있는 '9' 개의 입력 특징이 있습니다.

# 테이블을 빠르게 훑어 보시면서 일관성을 체크해보세요. 정상적으로 보입니까?

# 직접 데이터를 살펴보십시오. 다 괜찮아 보이시나요? 얼마나 많은 문제점을 발견 할 수 있는지 봅시다. 통계에 대한 배경 지식이 없어도 걱정하지 마십시오. 상식적이기만 하면 충분히 발견할 수 있습니다.

# 데이터를 직접 검토 한 후에는 데이터를 확인하는 방법에 대한 추가 고려 사항을 확인하십시오.

# 특히 latitude(위도) 및 longitude(경도) 두 가지 특징에 대해 자세히 살펴 보겠습니다. 이들은 해당 도시 블록의 지리적 좌표입니다.

# 이것으로 멋진 시각화를 만들 수 있습니다 - latitude 및 longitude를 그려보고 색상을 사용하여 median_house_value를 표시해 봅시다.

plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_examples["longitude"],
            validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

ax = plt.subplot(1,2,2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()
plt.show()

# 잠시만요... 샌프란시스코와 로스앤젤레스와 같이 값비싼 지역에 빨간색이 표시된, 캘리포니아 주에 대한 멋진 지도가 나왔어야 합니다.

# 훈련 세트의 결과는 실제지도 와 비교하면 그럭저럭 비슷하지만, 검증 데이터의 결과는 분명히 그렇지 않습니다.

# 다시 올라가서 일관성 체크 데이터를 다시 보십시오.

# 훈련 데이터와 검증 데이터간의 특징 또는 목표의 분포가 다른 점이 있습니까?

# 핵심 문제들을 알려면 솔루션에서 힌트를 확인하십시오. (솔루션을 추가할 예정입니다)

## 작업 1 : 데이터 가져 오기 및 전처리 코드로 돌아가서 버그가 있는지 확인하기

# 그렇다면 위에서 버그를 수정하십시오. 1~2분 이상 쓰지 마십시오. 버그를 찾을 수 없는 경우 솔루션에서 힌트를 확인하십시오. (솔루션을 추가할 예정입니다)

# 문제를 찾아서 고쳐봤을 때 위에 위도를 그리는 latitude / longitude 코드를 다시 실행하고 일관성 체크 결과가 나아지는지 확인하십시오.

# 그런데 여기서 중요한 교훈이 있습니다.

# ML에서의 디버깅은 종종 코드 디버깅보다는 데이터 디버깅입니다.

# 데이터가 잘못되면 가장 진보된 ML 코드조차도 아무것도 할 수 없습니다.

# 힌트: Pandas 강의의 reindex 부분을 살펴보세요.

#
# 여기에 코드를 작성하세요
#


## 작업 2 : 모델을 훈련시키고 평가하십시오.

# 다른 하이퍼 매개 변수 설정을 시도하는 데 5분 정도를 사용해 봅시다. 최대한의 검증 성능을 얻으세요.

# TensorFlow Estimators 라이브러리에서 제공하는LinearRegressor 인터페이스를 사용하여 linear_regressor를 설정하는 코드를 작성하십시오.

# 이전 연습의 코드를 사용하는 것은 괜찮습니다. 적절한 데이터 세트를 대상으로 fit()및 predict()를 호출 하고 싶을겁니다.

# 단일 특징 대신 여러 입력 특징을 사용하는 경우에는 특별한 함수가 필요하지 않습니다. Estimators 인터페이스에서 Pandas DataFrame 객체를 사용할 수 있습니다.

# DataFrame에 여러 특징이 정의되어있는 경우 (우리가 했던것과 마찬가지로) 그 모든 특징들이 사용됩니다.

# 훈련 데이터 및 검증 데이터의 손실을 비교하십시오.

# 단일 원시 특징에서 RMSE (root mean square error)는 약 180이었습니다.

# 여러 특징을 사용할 수 있게 된 지금 얼마나 더 나아졌는지 확인하십시오.

# 앞에서 살펴본 일관성 검사 방법 중 일부를 사용하십시오. 다음과 같은 방법들입니다.

 # 예상 분포와 실제 목표 값을 비교
  # 예상 값과 목표 값의 산점도 (scatter plot) 시각화
  # '위도'와 '경도'를 사용하여 검증 데이터의 두 가지 산점도 시각화
  # 실제 목표 인median_house_value를 보여주는 산점도
  # 비교를 위해 예측한 median_house_value에 색을 입혀 보여주는 산점도

#
# 여기에 코드를 작성하세요
#

## 작업 3 : 테스트 데이터 평가

# 아래 셀에서 테스트 데이터 세트를 불러와서 모델을 평가하십시오.

# 검증 데이터에 대해 많은 반복 작업을 수행했습니다. 그 결과 혹시 특정 표본 데이터의 특색에 지나치게 오버피팅되지 않았는지 확인합시다.

# 테스트 데이터 세트는 여기에 있습니다.

# 테스트 성능과 검증 성능을 비교하면 어떻습니까? 이 모델의 일반적인 성능에 대해 무엇을 알려주나요?

#
# 여기에 코드를 작성하세요
#


### 특정 집합 (Feature sets)
## 학습 목표

# 복잡한 특징 집합과 비교해도 비슷하게 잘 동작하는 가장 작은 특징 집합을 만들어 봅니다.
# 지금까지 우리는 모든 특징을 모델에 던졌습니다. 특징이 적은 모델은 리소스가 적고 유지 관리가 쉽습니다. 특징을 줄여봅시다.

# 이전과 마찬가지로 데이터를 불러오고 준비합시다.

import math

# from sorna.display import display
from IPython.display import display

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_io

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("http://datasets.lablup.ai/public/tutorials/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
  """캘리포니아 주거 데이터 세트로부터 입력 특징을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    모델에서 사용할 (합성 특징들을 포함한) 특징이 든 DataFrame.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # 합성 특징 만들기.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """캘리포니아 주거 데이터로부터 목표 특징 (레이블)들을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    목표 특징을 포함한 DataFrame.
  """
  output_targets = pd.DataFrame()
  # 천단위 달러로 목표 규모를 조정합니다.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
display(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
display(training_targets.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
display(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
display(validation_targets.describe())

## 작업 1 : 좋은 특징 집합 개발

# 단지 2 ~ 3가지 특징만 사용하여 얻을 수 있는 최고의 성능은 얼마일까요?

# 상관 행렬은 목표와 비교한 각 특징과, 다른 특징들과 비교한 각 특징의 쌍방향 상관 관계를 보여줍니다.

# 여기서 상관 관계는 피어슨 상관 계수로 정의됩니다. 이 코스를 위해 수학적 세부 사항을 이해할 필요는 없습니다.

# 상관 계수의 의미는 다음과 같습니다.

 # -1.0 : 완벽한 음의 상관 관계
 # 0.0 : 상관 관계 없음
 # 1.0 : 완전한 양의 상관 관계

correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]

display(correlation_dataframe.corr())

# 이상적으로, 우리는 목표와 강한 상관 관계가 있는 특징을 원합니다. 우리는 또한 너무 강하게 상호 연관되지 않아서 서로 독립적인 정보를 담는 특징들을 갖고 싶습니다.

# 이 정보를 사용하여 특징들을 제거하십시오. 두 가지 원시 특징의 비율과 같은 추가 합성 특징을 만들 수도 있습니다.

# 편의를 위해 이전 연습에서 작성한 코드를 아래에 추가했습니다

def train_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """선형 회귀 모델을 훈련합니다.

  이 함수는 훈련뿐만 아니라 시간 경과에 따른 훈련 및 검증 손실의 플롯 등 훈련 진행 정보도 보여줍니다.

  Args:
    learning_rate: `float`, 학습율.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    batch_size: 0이 아닌 `int`, 배치 크기.
    training_examples: 훈련을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    training_targets: 훈련을 위한 목표로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`
    validation_examples: 검증을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    validation_targets: 검증을 위한 목표로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`

  Returns:
    훈련 데이터로 훈련한 `LinearRegressor` 객체.
  """
  periods = 10
  steps_per_period = steps / periods

  # 선형 회귀 객체 생성.
  feature_columns = set([tf.contrib.layers.real_valued_column(my_feature) for my_feature in training_examples])
  linear_regressor = tf.contrib.learn.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0
  )

  # 입력 함수 생성.
  training_input_fn = learn_io.pandas_input_fn(
      x=training_examples, y=training_targets["median_house_value"],
      num_epochs=None, batch_size=batch_size)
  predict_training_input_fn = learn_io.pandas_input_fn(
      x=training_examples, y=training_targets["median_house_value"],
      num_epochs=1, shuffle=False)
  predict_validation_input_fn = learn_io.pandas_input_fn(
      x=validation_examples, y=validation_targets["median_house_value"],
      num_epochs=1, shuffle=False)

  # 모델을 훈련 시키되 루프 내부에서 수행하여 손실 매트릭을 주기적으로 평가할 수 있게 합니다.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # 이전 상태에서 시작하여 모델을 교육.
    linear_regressor.fit(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # 잠시 멈추고 예측을 계산합니다.
    training_predictions = list(linear_regressor.predict(input_fn=predict_training_input_fn))
    validation_predictions = list(linear_regressor.predict(input_fn=predict_validation_input_fn))
    # 훈련 및 검증 손실 계산.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # 주기적으로 현재의 손실을 출력.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # 이번 주기의 손실 매트릭을 리스트에 추가.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")


  # 주기에 따른 손실 매트릭 그래프 출력
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor

# 5분 동안 훌륭한 특징 및 교육 매개 변수 조합을 알아보세요. 그런 다음 솔루션을 선택하여 우리가 선택한 것을 확인하십시오. 다른 특징들은 다른 학습 매개 변수가 필요할 수 있음을 잊지 마십시오.

#
# 여기에 코드를 입력하세요: 당신이 고른 특징을 작은따옴표로 묶은 문자열 리스트로 추가하세요.
#
minimal_features = [
]

assert minimal_features, "You must select at least one feature!"

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

#
# 이 파라미터들을 수정하는 것을 잊지 마세요.
#
train_model(
    learning_rate=0.001,
    steps=500,
    batch_size=5,
    training_examples=minimal_training_examples,
    training_targets=training_targets,
    validation_examples=minimal_validation_examples,
    validation_targets=validation_targets)

## 작업 2 : 위도를 보다 잘 활용하기

# 위도와 더 잘 작동하는 합성 특징을 만들어보십시오.

# 위도 (latitude)와 median_house_value를 함께 그려 보면 실제로 거기에는 선형 관계가 없다는 것을 알 수 있습니다.

# 대신 로스앤젤레스와 샌프란시스코에 대략 일치하는 몇 가지 봉우리가 있습니다.

plt.figure()
plt.scatter(training_examples["latitude"], training_targets["median_house_value"])
plt.show()

# 예를 들어 다음과 같은 방법들이 가능합니다.

 # 상관 행렬을 통해 상관계수를 살펴보고 새로운 변수들을 추가할 수 있습니다.
 # 샌프란시스코의 latitude는 37.773972임을 이용하여 lat_dist_from_SF = |latitude − 37.773972| 라는 새로운 특징을 만들어 낼 수 있습니다.
 # latitude를 |latitude - 38|값으로 매핑하는 특징을 만들 수 있고, 이것을distance_from_san_francisco라고 부를 수도 있습니다.
 # latitude_32_to_33,latitude_33_to_34와 같이 latitude/logitude 를 10등분 하여 새로운 특징들을 만들어 해당 범위 안에 있으면 1 아니면 0 값으로 나타 낼 수도 있습니다.

# 이외에도 많은 방법들이 있습니다. 그동안 익힌 방법들을 사용해 최고의 검증 성능을 구해보세요. 상관 행렬을 사용하여 어떤 방법을 사용할지 계획해 보세요. 좋아 보이는 것을 발견하면 모델에 추가하십시오.

# 저는 약 68.48495469688254을 얻었습니다. 당신이 얻을 수있는 최고의 검증 성능은 무엇입니까?

#
# 여기에 코드를 입력하세요: latutide를 이용한 합성 특징을 포함한 새 데이터 세트로 훈련시켜보세요.
#


### 특징 조합 (feature crosses)
# 학습 목표

 # 추가 합성 기능을 추가하여 선형 회귀 모델을 개선합니다 (이전 연습의 연속입니다)
  # * 입력 함수를 사용하여 pandasDataFrame 객체를 Tensors로 변환하고 fit() 및 predict()연산에서 입력 함수를 호출합니다.
 # 모델 훈련에 FTRL 최적화 알고리즘을 사용합니다
 # one-hot 인코딩, 비닝 (binning) 및 특징 조합(feature crosses)을 통해 새로운 합성 기능을 생성합니다.

# 먼저, 입력을 정의하고 데이터 불러오기 코드를 작성해 봅시다.

import math

# from sorna.display import display
from IPython.display import display

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataframe = pd.read_csv("http://datasets.lablup.ai/public/tutorials/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
  """캘리포니아 주거 데이터 세트로부터 입력 특징을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    모델에서 사용할 (합성 특징들을 포함한) 특징이 든 DataFrame.
  """
  selected_features = california_housing_dataframe[
    ["latitude",
     "longitude",
     "housing_median_age",
     "total_rooms",
     "total_bedrooms",
     "population",
     "households",
     "median_income"]]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """캘리포니아 주거 데이터로부터 목표 특징 (레이블)들을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    목표 특징을 포함한 DataFrame.
  """
  output_targets = pd.DataFrame()
  # 천단위 달러로 목표 규모를 조정합니다.
  output_targets["median_house_value"] = (
    california_housing_dataframe["median_house_value"] / 1000.0)
  return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
display(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
display(training_targets.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
display(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
display(validation_targets.describe())

## 특징 엔지니어링

# 잘 맞는 특징을 만들면 특히 회귀와 같은 간단한 모델의 경우 ML 모델이 크게 향상됩니다. 우리는 이전 연습에서 2개 (또는 그 이상)의 독립적인 특징은 종종 특징으로부터 파생된 특징만큼 많은 정보를 제공하지 않는다는 것을 알게 되었습니다.

# 앞의 예에서 이미 합성 기능을 사용했습니다 :rooms_per_person.

# 특정 열에 대한 연산을 수행하여 간단하게 합성 특징을 생성 할 수 있습니다. 그러나 버킷화 또는 조합 버킷 특징과 같은 복잡한 작업에서는 장황할 수 있습니다. 특징 열은 합성 특징을 쉽게 추가 할 수 있게 하는 강력한 추상화입니다.

longitude = tf.contrib.layers.real_valued_column("longitude")
latitude = tf.contrib.layers.real_valued_column("latitude")
housing_median_age = tf.contrib.layers.real_valued_column("housing_median_age")
households = tf.contrib.layers.real_valued_column("households")
median_income = tf.contrib.layers.real_valued_column("median_income")
rooms_per_person = tf.contrib.layers.real_valued_column("rooms_per_person")

feature_columns = set([
    longitude,
    latitude,
    housing_median_age,
    households,
    median_income,
    rooms_per_person])

## 입력 함수

# 이전에는 Pandas의 DataFrame 객체를 사용하여 데이터를 예측기(estimator)에 전달했습니다. 더 유연하지만 더 복잡한 데이터 전달 방법은 입력 함수를 사용하는 것입니다.

# estimators API의 특이한 점 중 하나는 입력 함수가 데이터를 배치로 분할하는 책임이 있으므로 input_fn 을 사용할 때batch_size 인수가 무시된다는 것입니다. 배치 크기는 입력 함수가 반환하는 행 수에 따라 결정됩니다 (아래 참조).

# 입력 함수는 TensorFlow에서 사용되는 핵심 데이터 유형인 Tensor 객체를 반환합니다. 더 구체적으로 말하면, 입력 함수는 다음과 같은(features, label)튜플을 반환해야 합니다 :

 # features : 모양(n, 1) 의 Tensor 값에 string 값 (피쳐 이름)을 매핑하는 dict. n은 입력 함수에 의해 반환된 데이터 행의 수 (따라서 배치 크기)입니다.
 # label : 대응하는 레이블을 나타내는 모양. (n, 1) 의 Tensor.

# 참고로, 입력 함수들은 대개 순차적으로 데이터를 읽는 대기열을 생성하지만 여기서는 다루지 않는 고급 주제입니다. 데이터가 너무 커서 메모리에 미리 로드 할 수없는 경우에 필요합니다.

# 간결함을 위해, 우리 함수는 전체 DataFrame 을 Tensor 로 변환 할 것입니다. 이것은 일괄 처리 크기를 12000(그리고 유효성 검사를 위해 각각 5000) - 다소 큰 크기로 사용함을 의미합니다만, 우리의 작은 모델에서는 잘 작동 할 것입니다. 이렇게 하면 훈련 속도가 약간 느려지지만 벡터 최적화 덕분에 성능 저하가 그렇게 심하지는 않을 것입니다.

# 여기에 필요한 입력 함수는 다음과 같습니다.

def input_function(examples_df, targets_df, single_read=False):
  """예제/목표 쌍 `DataFrame` 을`Tensor`로 변환합니다.

  `Tensor`는 `(N,1)` 형태로 변형됩니다. `N`은 `DataFrame`에 들어있는 예제의 수입니다.

  Args:
    examples_df: 입력 특징을 포함하는 `DataFrame`. 모든 열은 해당 입력 특징 `Tensor`객체로 변환됩니다.
    targets_df: `examples df`의 각 예제에 해당하는 단일 열을 포함한 `DataFrame`.
    single_read: 이 함수가 dataset을 한 번 읽은 후에 멈춰야 하는지의 여부를 가리키는 `bool`. 
      `False`인 경우, 데이터 세트를 반복 순환합니다. 이 중지 메커니즘은 estimator의 `predict()`가 
      읽는 값의 수를 제한하기 위해 사용됩니다.

Returns:
    `(input_features, target_tensor)` `tuple`:
      input_features: 특징의 열 이름 문자열 값을 실제 특징 값의 `Tensor`에 매핑하는 `dict`.
      target_tensor: 목표 값을 나타내는 `Tensor`.
  """
  features = {}
  for column_name in examples_df.keys():
    batch_tensor = tf.to_float(
        tf.reshape(tf.constant(examples_df[column_name].values), [-1, 1]))
    if single_read:
      features[column_name] = tf.train.limit_epochs(batch_tensor, num_epochs=1)
    else:
      features[column_name] = batch_tensor
  target_tensor = tf.to_float(
      tf.reshape(tf.constant(targets_df[targets_df.keys()[0]].values), [-1, 1]))

  return features, target_tensor

# 예를 들어, 아래 코드는 캘리포니아 주택 데이터 세트의 몇 가지 샘플 레코드를 전달했을 때 입력 함수의 출력을 보여줍니다.

# 이 코드는 오직 설명을 위한 것입니다. 모델을 교육하는 데 반드시 필요한 것은 아닙니다만, 다양한 특징 의 효과를 시각화하는 것은 유용함을 알게 될 것입니다.

def sample_from_input_function(input_fn):
  """주어진 입력 함수에서 몇개의 샘플을 반환합니다.

  Args:
    input_fn: `Estimator`의 입력함수 조건을 충족하는 입력함수.
  Returns:
    이 함수에서 반환하는 적은 수의 레코드를 포함한 `DataFrame`.
  """

  examples, target = input_fn()

  example_samples = {
    name: tf.strided_slice(values, [0, 0], [5, 1]) for name, values in examples.items()
  }
  target_samples = tf.strided_slice(target, [0, 0], [5, 1])

  with tf.Session() as sess:
    example_sample_values, target_sample_values = sess.run(
        [example_samples, target_samples])

  results = pd.DataFrame()
  for name, values in example_sample_values.items():
    results[name] = pd.Series(values.reshape(-1))
  results['target'] = target_sample_values.reshape(-1)

  return results

samples = sample_from_input_function(
  lambda: input_function(training_examples, training_targets))
display(samples)

## FTRL 최적화 알고리즘

# 고차원 선형 모델은 FTRL이라는 그래디언트 기반 최적화 변형을 사용하면 이득을 얻습니다. 이 알고리즘은 다른 계수에 대해 학습 속도를 다르게 조정할 수있는 이점이 있습니다. 이는 일부 특징이 0이 아닌 값을 거의 사용하지 않는 경우 유용 할 수 있습니다. (또한 L1 정규화와 잘 어울립니다.) FtrlOptimizer를 사용하여 FTRL을 적용 할 수 있습니다.

def train_model(
    learning_rate,
    steps,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """선형 회귀 모델을 훈련합니다.

  이 함수는 훈련뿐만 아니라 시간 경과에 따른 훈련 및 검증 손실의 플롯 등 훈련 진행 정보도 보여줍니다.

  Args:
    learning_rate: `float`, 학습율.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    feature_columns: 사용할 입력 특징 열을 지정하는 `set`.
    training_examples: 훈련을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    training_targets: 훈련을 위한 목표로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`
    validation_examples: 검증을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    validation_targets: 검증을 위한 목표로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`

  Returns:
    훈련 데이터로 훈련한 `LinearRegressor` 객체.
  """

  periods = 10
  steps_per_period = steps / periods

  # 선형 회귀 객체 생성.
  linear_regressor = tf.contrib.learn.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=tf.train.FtrlOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0
  )

  training_input_function = lambda: input_function(
      training_examples, training_targets)
  training_input_function_for_predict = lambda: input_function(
      training_examples, training_targets, single_read=True)
  validation_input_function_for_predict = lambda: input_function(
      validation_examples, validation_targets, single_read=True)

  # 모델을 훈련 시키되 루프 내부에서 수행하여 손실 매트릭을 주기적으로 평가할 수 있게 합니다.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # 이전 상태에서 시작하여 모델을 교육.
    linear_regressor.fit(
        input_fn=training_input_function,
        steps=steps_per_period
    )
    # 잠시 멈추고 예측을 계산합니다.
    training_predictions = list(linear_regressor.predict(
        input_fn=training_input_function_for_predict))
    validation_predictions = list(linear_regressor.predict(
        input_fn=validation_input_function_for_predict))
    # 훈련 및 검증 손실 계산.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # 주기적으로 현재의 손실을 출력.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # 이번 주기의 손실 매트릭을 리스트에 추가.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")


  # 주기에 따른 손실 매트릭 그래프 출력
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout(True)
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()
  return linear_regressor

_ = train_model(
    learning_rate=1.0,
    steps=500,
    feature_columns=feature_columns,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 이산 특징을 위한 one-hot 인코딩

# 이산 (즉, string(문자열), enumerations(열거), integers(정수)) 특징은 일반적으로 로지스틱 회귀 모델을 훈련하기 전에 이진 특징으로 변환합니다.

# 예를 들어, 0,1 또는 2 값 중 하나를 취할 수있는 합성 피쳐를 만들었고 몇 가지 훈련 포인트가 있다고 가정합시다.

# 가능한 범주 별 값에 대해 실제 값의 새로운 이진 특징을 만듭니다.이 값은 두 가지 값 중 하나를 취할 수 있습니다 : 예제에 해당 값이 있으면 1.0을, 그렇지 않으면 0.0으로 만듭니다. 위의 예에서 범주형 특징은 세 가지 특징으로 변환되며 훈련용은 다음과 같이 보일겁니다.

## 버킷화된 (binned) 특징

# 버킷 화는 비닝(binning)이라고도합니다.

# 우리는 다음과 같은 3 가지 버킷에 인구를 버킷화할 수 있습니다 (예를 들면):
 # - bucket_0 (<5000) : 인구가 적은 블록에 해당
 # - bucket_1 (5000 - 25000) : 인구가 적당한 블록에 해당
 # - bucket_2 (> 25000) : 인구가 많은 블록에 해당

# 앞의 버킷 정의가 주어지면 다음과 같은population 벡터:

 # [[10001], [42004], [2500], [18000]]
# 는 아래의 버킷화 된 특징 벡터가 됩니다:
 # [[1], [2], [0], [1]]

# 이제 특징 값이 버킷 인덱스입니다. 이러한 인덱스들은 이산 특징으로 간주됩니다. 전형적으로, 이러한 인덱스들은 one-hot 표현으로 더 변환됩니다. (여기서는 직관적으로 의미가 보이도록 버킷화된 특징 벡터를 바로 사용하겠습니다.)

# 버킷화 된 특징을 정의하려면 각 버킷을 구분하는 경계를 담는 bucketized_column 을 사용하십시오. 아래 셀의 함수는 변위(quantile)에 기초하여 이 경계를 계산하므로 각 버킷에는 같은 수의 요소가 담깁니다.

def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

# Divide households into 7 buckets.
bucketized_households = tf.contrib.layers.bucketized_column(
  households, boundaries=get_quantile_based_boundaries(
    california_housing_dataframe["households"], 7))

# Divide longitude into 10 buckets.
bucketized_longitude = tf.contrib.layers.bucketized_column(
  longitude, boundaries=get_quantile_based_boundaries(
    california_housing_dataframe["longitude"], 10))

## 작업 1 : 버킷화된 특징 열로 모델을 훈련시키기

# 예제에서 실수값인 모든 특징들을 버킷으로 만들어 모델을 훈련시키고 결과가 개선되는지 확인하십시오.

# 앞의 코드 블록에서 두 개의 실수값 열 (즉, '세대(household)'와 '경도(longitude)')이 버킷화 된 특징 열로 변환되었습니다. 당신의 임무는 나머지 열을 버킷화하고 코드를 실행하여 모델을 훈련시키는 것입니다. 버킷의 범위를 찾는 다양한 경험적 방법이 있습니다. 이 연습에서는 각 양동이가 같은 수의 예제를 갖도록 양각 경계를 선택하는 quantile 기반 기술을 사용합니다.

#
# 여기에 코드를 입력하세요: 위의 예들을 따라서 아래의 열들을 버킷화하세요.
#
bucketized_latitude = 
bucketized_housing_median_age = 
bucketized_median_income =
bucketized_rooms_per_person =

bucketized_feature_columns=set([
  bucketized_longitude,
  bucketized_latitude,
  bucketized_housing_median_age,
  bucketized_households,
  bucketized_median_income,
  bucketized_rooms_per_person])

_ = train_model(
    learning_rate=1.0,
    steps=500,
    feature_columns=bucketized_feature_columns,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 특징 조합

# 두 개 (또는 그 이상)의 특징을 조합하는 것은 선형 모델을 사용하여 비선형 관계를 학습하는 영리한 방법입니다. 우리 문제에서 학습을 위해 위도(latitude)라는 특징을 사용하는 경우, 모델은 특정 위도 (또는 도시 블록을 버킷화했다면 특정 양의 위도 범위)에서 다른 블록보다 비싸기 쉽다는 것을 알 수 있습니다. '경도' 특징과 비슷하지요. 그러나 '경도' 와 '위도'를 조합하면, 조합한 특징은 더 잘 정의 된 도시 블록을 나타냅니다. 모델이 (위도와 경도의 범위 내의)특정 도시 블록이 다른 블록보다 더 비싸다는 것을 알게 된다면, 두 특징이 개별적으로 고려되는 경우보다 훨씬 강력한 신호입니다.

# 현재 특징 열 API는 조합에 대한 개별 기능 만 지원합니다. '위도'또는 '경도'와 같은 두 개의 연속 값을 조합할 때 버킷화를 할 수 있습니다.

# 위도latitude와 경도longitude기능을 조합하면 (예 : longitude가 '2'버킷으로 버킷화되고 latitude가 '3'버킷), 6개의 조합된 이진 기능이 생깁니다. 모델을 교육할 때 각 기능은 별도의 가중치를 갖게 될 것입니다.

## 작업 2 : 특징 조합을 사용하여 모델 학습 시키기

# '경도longitude'와 '위도latitude' 특징을 조합한 특징을 모델에 추가하고, 훈련하고, 결과가 개선되는지 확인하세요.

long_x_lat = tf.contrib.layers.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

#
# Your code here: Create a feature column set that includes the cross.
# 여기에 코드를 입력하세요: 조합을 포함한 특징 열 집합을 만드세요.
#
feature_columns_with_cross = 

_ = train_model(
    learning_rate=1.0,
    steps=500,
    feature_columns=feature_columns_with_cross,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 추가 도전: 더 많은 합성 기능을 시도해보십시오.

# 지금까지 간단한 버킷화 된 열과 특징 조합을 시도했습니다만, 잠재적으로 결과를 향상시킬 수있는 더 많은 조합이 있습니다. 예를 들어 여러 열을 조합할 수 있습니다. 버킷 수를 변경하면 어떨까요? 다른 합성 특징들은 어떤 것들이 가능할까요? 그 특징들이 모델을 개선할까요?