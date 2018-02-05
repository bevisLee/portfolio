### 인공신경망 기초

## 학습 목표

# TensorFlow DNNRegressor 클래스를 사용하여 신경망(Neural Network, NN)과 은닉층(Hidden Layer)을 정의합니다.
 # * 데이터 집합의 비선형성을 배우고 선형 회귀 모델보다 우수한 성능을 달성하도록 신경망을 훈련시킵니다.
# 이전 연습에서는 모델에 비선형성을 적용 할 수 있도록 합성 기능을 사용했습니다. 중요한 비선형성 하나는 위도와 경도 관련한 값이었지만 다른 비선형성이 있을 수 있습니다.

# 또한 이전 실습의 로지스틱 회귀 작업 대신 표준 회귀 작업으로 다시 돌아가겠습니다. 즉,median_house_value를 직접 예측할 것입니다.

# 먼저, 데이터를 불러오고 준비해봅시다.

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
  # Scale the target to be in units of thousands of dollars.
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

## 신경망 구축하기

# NN은 DNNRegressor 클래스에 의해 정의됩니다.
 # DNNRegressor - https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNRegressor

# hidden_units를 사용하여 NN의 구조를 정의하십시오. hidden_units 인자는 int의 리스트를 제공합니다. 각 int는 은닉층에 해당하며 그 안에 있는 노드의 수를 나타냅니다. 예를 들어 아래의 할당을 봅시다.

# hidden_units=[3,10]

# 앞의 할당은 두 개의 은닉층이 있는 신경망을 지정합니다.

# 첫 번째 은닉층은 3 개의 노드를 포함합니다.
# 두 번째 은닉층은 10 개의 노드를 포함합니다.
# 더 많은 계층을 추가하려면 목록에 int를 더 추가해야합니다. 예를 들어,hidden_units = [10,20,30,40]은 각각 10, 20, 30, 40 단위로 4개의 계층을 생성합니다.

# 기본적으로 모든 은닉층은 ReLu 활성화 함수를 사용하며 모두 서로 연결되어 있습니다.

def train_nn_regression_model(
    learning_rate,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """신경망 회귀 모델을 훈련합니다.

  이 함수는 훈련뿐만 아니라 시간 경과에 따른 훈련 및 검증 손실의 플롯 등 훈련 진행 정보도 보여줍니다.

  Args:
    learning_rate: `float`, 학습율.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    batch_size: 0이 아닌 `int`, 배치 크기.
    hidden_units: int 값의 `list`. 각 층 내의 뉴런수.
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
  dnn_regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_columns,
      hidden_units=hidden_units,
      optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0
  )

  # 입력 함수 만들기
  training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets["median_house_value"],
     num_epochs=None, batch_size=batch_size)
  predict_training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets["median_house_value"],
     num_epochs=1, shuffle=False)
  predict_validation_input_fn = learn_io.pandas_input_fn(
      x=validation_examples, y=validation_targets["median_house_value"],
      num_epochs=1, shuffle=False)

  # 모델을 훈련 시키되 루프 내부에서 수행하여 손실 매트릭을 주기적으로 평가할 수 있습니다.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # 이전 상태에서 시작하여 모델을 교육.
    dnn_regressor.fit(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # 잠시 멈추고 예측을 계산합니다.
    training_predictions = list(dnn_regressor.predict(input_fn=predict_training_input_fn))
    validation_predictions = list(dnn_regressor.predict(input_fn=predict_validation_input_fn))
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
  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor

## 작업 1 : NN 모델 훈련하기

# RMSE를 110보다 낮게 하는 것을 목표로 하이퍼 파라미터를 조정하십시오.

# 다음 블록을 실행하여 NN 모델을 학습하십시오.

# 많은 특징을 사용하는 선형 회귀 연습에서 110 정도의 RMSE가 꽤 좋았던 점을 기억합시다. 그것을 이기는 것이 목표입니다.

# 여기에서 수행할 작업은 다양한 학습 설정을 수정하여 유효성 검사 데이터 대상으로 정확성을 향상시키는 것입니다.

# 오버피팅(overfitting)은 NN의 실제 잠재적 위험입니다. 훈련 데이터 손실과 유효성 검증 데이터의 손실 사이의 격차를 살펴봄으로써 모델이 오버피팅되기 시작했는지 판단 할 수 있습니다. 그 격차가 커지기 시작하면, 그것은 보통 오버피팅의 신호입니다.

# 매우 다양한 설정이 가능하기 때문에 개발 과정을 위해 각 시도마다 메모를 작성하는 것이 좋습니다.

# 또한 좋은 설정값을 얻었으면 여러 번 실행 해보고 결과가 얼마나 반복적인지 확인하십시오. NN 가중치는 일반적으로 작은 무작위 값으로 초기화되므로 실행 간 차이점을 확인해야합니다.

dnn_regressor = train_nn_regression_model(
    learning_rate=0.01,
    steps=500,
    batch_size=10,
    hidden_units=[10, 2],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 작업 2 : 테스트 데이터 평가

# 검증 성능 결과가 테스트 데이터에서도 같게 나타나는지 확인하십시오.

# 만족스러운 모델을 얻은 후에는 테스트 데이터를 대상으로 돌려 검증 성능과 비교하십시오.

# 데이터는 여기에 있습니다.
 # data - C:\Users\bevis\Documents\Visual Studio 2017\Projects\Python_project\california_housing_train.csv

#
# 여기에 코드를 작성하세요.
#


### 인공신경망 개선하기

## 학습 목표

# 특징을 정규화하고 다양한 최적화 알고리즘을 적용하여 신경망의 성능을 향상시킵니다.
# 참고 :이 연습에서 설명한 최적화 방법은 신경망에만 국한되지는 않으며 대부분의 유형의 모델을 개선하는 효과적인 방법입니다.

# 먼저 데이터를 불러옵니다.

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
  """캘리포니아 주거 데이터로부터 타겟 특징 (레이블)들을 준비합니다.

  Args:
    california_housing_dataframe: 캘리포니아 주거 데이터 세트의 데이터가 든 Pandas DataFrame
  Returns:
    타겟 특징을 포함한 DataFrame.
  """
  output_targets = pd.DataFrame()
  # 타겟을 1000달러 단위로 스케일.
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

# 이제 우리 신경망을 훈련시킵시다.

def train_nn_regression_model(
    optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """신경망 회귀 모델을 훈련합니다.

  이 함수는 훈련에 더하여 훈련 진행 정보 및 시간 경과에 따른 훈련 및 검증 손실의 플롯을 보여줍니다.

  Args:
    optimizer: `tf.train.Optimizer` 인스턴스, 사용할 optimizer.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    batch_size: 0이 아닌 `int`, 배치 크기.
    hidden_units: int 값의 `list`. 각 층 내의 뉴런수.
    training_examples: 훈련을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    training_targets: 훈련을 위한 타겟으로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`
    validation_examples: 검증을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    validation_targets: 검증을 위한 타겟으로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`

  Returns:
    훈련 데이터로 훈련한 `LinearRegressor` 객체.
    `(estimator, training_losses, validation_losses)` tuple:
      estimator: 훈련된 `DNNRegressor` 객체.
      training_losses: 훈련 중의 훈련 손실 값들이 든 `list`
      validation_losses: 검증 중의 검증 손실 값들이 든 `list`
  """


  periods = 10
  steps_per_period = steps / periods

  # 선형 회귀 객체 생성.
  feature_columns = set([tf.contrib.layers.real_valued_column(my_feature) for my_feature in training_examples])
  dnn_regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_columns,
      hidden_units=hidden_units,
      optimizer=optimizer,
      gradient_clip_norm=5.0
  )

  # 입력 함수 생성
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
    dnn_regressor.fit(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # 잠시 멈추고 예측을 계산합니다.
    training_predictions = list(dnn_regressor.predict(input_fn=predict_training_input_fn))
    validation_predictions = list(dnn_regressor.predict(input_fn=predict_validation_input_fn))
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
  plt.clf()
  plt.close()
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout(True)
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor, training_rmse, validation_rmse

_ = train_nn_regression_model(
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 선형 스케일링

# 입력 값을 -1, 1 범위로 정규화하는 것이 좋은 표준 사례일 수 있습니다. 이렇게하면 SGD가 한 차원에서 너무 큰 단계를 수행하거나 다른 차원에서 너무 작은 단계를 수행하는 것을 막는데 도움이됩니다. 수치 최적화 팬들은 여기에 예조건기(preconditioner)를 사용하는 아이디어를 연관지을 수 있음을 알 수도 있습니다.

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

## 작업 1 : 선형 스케일링을 사용하여 특징을 표준화하기

# 척도 -1, 1로 입력을 표준화하십시오.

# 약 5 분을 훈련하고 새로 표준화 된 데이터를 평가합니다. 얼마나 잘 할 수 있습니까?

# 경험적으로 신경망은 입력 특징들이 대략 동일한 규모일 때 가장 잘 훈련되어 있습니다.

# 표준화 된 데이터를 점검하십시오. (만약 하나의 특징을 표준화하는 것을 잊었다면 어떻게 될까요?)

def normalize_linear_scale(examples_dataframe):
  """모든 특징이 선형적으로 정규화된 입력 `DataFrame`을 반환합니다."""
  #
  # 여기에 코드를 입력하세요: 입력을 정규화하세요.
  #
  pass

normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

_ = train_nn_regression_model(
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

## 작업 2 : 다른 최적화 도구를 사용해보십시오.

# Adagrad 및 Adam 최적화 도구를 사용하고 성능을 비교하십시오.

# Adagrad 최적화 도구가 하나의 대안입니다. Adagrad의 핵심 통찰력은 모델의 각 계수에 대해 학습 속도를 적응적으로 수정하여 실질 학습 속도를 단조 감소시키는 것입니다. 이는 볼록(convex) 문제에 대해서는 효과적이지만 비볼록(non-convex) 문제인 신경망 (Neural Net) 훈련에 항상 적합한 것은 아닙니다. GradientDescentOptimizer 대신AdagradOptimizer를 지정하여 Adagrad를 사용할 수 있습니다. Adagrad에서는 보다 큰 학습률을 사용해야 할 수도 있습니다.

# 비볼록 최적화 문제의 경우 Adam은 때때로 Adagrad보다 효율적입니다. Adam을 사용하려면,tf.train.AdamOptimizer 메소드를 호출하십시오. 이 방법은 몇 가지 선택적 하이퍼 매개 변수를 인수로 사용하지만, 우리의 솔루션은 이들 중 하나만 지정합니다 (learning_rate). 프로덕션 환경에서는 선택적 하이퍼 매개 변수를 매우 신중하게 지정하고 조정해야합니다.

#
# 여기에 코드를 입력하세요: Adagrad로 네트워크를 다시 학습시켜본 후 Adam으로도 해 보세요.
#

## 작업 3 : 대체 정규화 방법을 알아보기

# 성능을 더욱 향상 시키려면 다양한 특징에 대한 대체 정규화를 시도하십시오.

# 변형된 데이터에 대한 온전성 체크 요약 통계를 면밀히 살펴보면 일부 특징들의 경우엔 선형 스케일링후 '-1'에 가깝게 집계됨을 알 수 있습니다.

# 예를 들어, 많은 특징들은 '0.0'대신에 '-0.8'정도의 중간값을 갖는 경우가 있습니다.

plt.clf()
plt.close()
_ = training_examples.hist(bins = 20, figsize = (18, 12), xlabelsize = 2)
plt.show()

# 이러한 특징을 변형 할 수 있는 추가 방법을 선택함으로써 더 나은 결과를 얻을 수 있습니다.

# 예를 들어, 로그 스케일링은 일부 특징을 개선할 수 있습니다. 또는 극단값들을 잘라내면 나머지 데이터들의 스케일을 통해 더 많은 정보를 얻을 수 있습니다.

def log_normalize(series):
  return series.apply(lambda x:math.log(x+1.0))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(
    min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))

# 위의 블록에는 가능한 추가 정규화 기능이 몇 가지 포함되어 있습니다. 이 중 일부를 실행해 보거나 직접 추가하세요.

# 타겟을 정규화한다면, 손실 메트릭을 비교할 수 있도록 예측들을 비정규화해야 합니다.

def normalize(examples_dataframe):
  """모든 특징이 정규화된 입력 `DataFrame`을 반환합니다."""
  #
  # 여기에 코드를 작성하세요. 입력을 정규화하세요.
  #
  pass

normalized_dataframe = normalize(preprocess_features(california_housing_dataframe))
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

_ = train_nn_regression_model(
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=normalized_training_examples,
    training_targets=training_targets,
    validation_examples=normalized_validation_examples,
    validation_targets=validation_targets)

## 선택 도전 : 위도 및 경도 특징만 사용하기

# 위도와 경도만 특징으로 사용하는 NN 모델을 교육해봅시다

# 부동산 중개인은 위치가 주택 가격의 유일한 중요한 특징이라고 말하는 것을 좋아합니다. 이것이 정말인지 확인하십시오.

# NN이 위도와 경도에서 복잡한 비선형성을 배울 수 있는 경우에만 잘 작동할 것입니다. 이전에 연습했던 것보다 더 많은 층을 가진 네트워크 구조가 필요할 수도 있습니다.

#
# 여기에 코드를 작성하세요. 위도 및 경도만 사용해서 네트워크를 훈련하세요.
#

### 신경망으로 손글씨 분류하기

## 학습 목표

# 선형 모델과 신경망을 모두 훈련시켜 고전적인 MNIST 데이터 세트의 손으로 쓴 숫자를 분류합니다.
# 선형 및 신경망 분류 모델의 성능을 비교합니다.
 # * 신경망 은닉층의 가중치를 시각화합니다.
# 목표는 각 입력 이미지를 올바른 숫자로 매핑하는 것입니다. 은닉층 몇 개와, 가장 알맞는 종류를 선택하는 소프트맥스 층이 있는 NN을 만들것입니다.

# 먼저 TensorFlow 및 기타 유틸리티로 데이터를 불러옵시다. 이 데이터는 원래의 MNIST 훈련 데이터의 샘플입니다. 무작위로 20000개의 행을 가져 왔습니다.

import subprocess
import pathlib

subprocess.run('wget http://datasets.lablup.ai/public/tutorials/mnist_train_small.csv -O /tmp/mnist_train_small.csv', shell=True)
filesize = pathlib.Path('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/mnist_train_small.csv').stat().st_size
print(f'Downloaded mnist_train_small.csv ({filesize:,} bytes)')

import glob
import io
import math
import os

# from sorna.display import display
from IPython.display import display

from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(
  io.open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/mnist_train_small.csv", "r"),
  sep=",",
  header=None)

mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
display(mnist_dataframe.head())

# 첫 번째 열에는 클래스 레이블이 들어있습니다. 나머지 열은 28×28=784픽셀 값에 대해 픽셀 당 하나의 특징값을 담고 있습니다. 이 784픽셀 값의 대부분은 0입니다. 그들이 모두 0이 아니라는 것을 확인하기 위해서는 시간을 들여야 할 수도 있습니다.

# 이 예는 상대적으로 낮은 해상도의 손으로 쓴 숫자의 고대비 이미지입니다. 0-9의 10 자리 숫자가 각각 표시되며 각 가능한 숫자에 고유한 클래스 레이블이 지정되어 있습니다. 따라서 이것은 10개의 클래스를 가진 다중 클래스 분류 문제입니다.

# 이제 레이블과 특징을 분석하고 몇 가지 예제를 온전성 검사로 살펴 보겠습니다. 이 데이터 세트에 머릿행(header)이 없으므로 원래 위치를 기반으로 컬럼을 추출 할 수 있게 해주는 iloc 을 사용합니다.

def parse_labels_and_features(dataset):
  """레이블 및 특징을 추출합니다.

  필요한 경우 이 함수에서 특징을 스케일하거나 변형하면 됩니다.

  Args:
    dataset: 첫 열에는 레이블이, 나머지 열에는 흑백 픽셀값이 든, 행 기준으로 정렬된 Pandas `Dataframe`.
  Returns:
    `(labels, features)` `tuple`:
      labels: Pandas `Series`.
      features: Pandas `DataFrame`.
  """
  labels = dataset[0]

  # DataFrame.loc 인덱스 범위는 양쪽 끝을 포함합니다.
  # iloc은 위치 기반임으로 파이썬과 동일하게 오른쪽 인덱스를 포함하지 않습니다.
  # loc은 레이블 기반임으로 1:784 은 [1, 2, ..., 784] 로 인식합니다.
  #features = dataset.loc[:,1:784]
  features = dataset.iloc[:, 1:]

  # 최댓값인 255로 나눠서 데이터를 [0, 1] 범위로 스케일합니다.
  features = features / 255

  return labels, features

# NOTE - iloc과 loc의 차이를 알아봅시다.

import pandas as pd
df = pd.DataFrame({"a": ['a', 'b', 'c'], 
    "b": [1, 2, 3], 
    "c": ["ᄀ", "ᄂ", "ᄃ"]})
print(df)

print(df.iloc[:, 1:3])

print(df.loc[:, 1:3])  # 1,2,3이라는 레이블이 없으므로 에러가 발생할 것입니다.

print(df.loc[:, ['a', 'c']]) # ok

training_targets, training_examples = parse_labels_and_features(mnist_dataframe.head(15000))
display(training_examples.describe())

validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe.tail(5000))
display(validation_examples.describe())

# 무작위로 추출한 예제와 해당 레이블을 봅시다.

rand_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.ix[rand_example].values.reshape(28, 28))
ax.set_title("Label: %i" % training_targets.ix[rand_example])
ax.grid(False)
plt.show()

## 작업 1 : MNIST 선형 모델 만들기

# 먼저 비교할 기준 모델을 만듭니다. LinearClassifier는 k 클래스마다 하나씩 총 k 개의 one-vs-all 분류자를 제공합니다.

# 정확도를 확인하고 시간이 지남에 따라 로그 손실을 시각화하는 것 외에도 혼동 행렬을 확인해 볼 수 있습니다. 혼동 행렬은 어떤 클래스가 다른 클래스로 잘못 분류되었는지 보여줍니다. 어떤 숫자들끼리 서로 잘못 분류되었습니까?

# 또한 우리는log_loss 함수를 사용하여 모델의 에러를 추적합니다. 훈련에 사용되는 LinearClassifier 내부의 손실 함수와 혼동되어서는 안됩니다.

def create_training_input_fn(features, labels, batch_size):
  """ 훈련을 위한 mnist 데이터를 estimator로 보내는 사용자 input_fn

  Args:
    features: 훈련 특징들.
    labels: 훈련 레이블들.
    batch_size: 훈련동안 사용할 배치 크기.

  Returns:
    훈련시 사용할 훈련 특징 및 레이블들로 된 배치를 반환하는 함수
  """
  def _input_fn():
    raw_features = tf.constant(features.values)
    raw_targets = tf.constant(labels.values)
    dataset_size = len(features)

    return tf.train.shuffle_batch(
        [raw_features, raw_targets],
        batch_size=batch_size,
        enqueue_many=True,
        capacity=2 * dataset_size,  # min_after_dequeue 보다 커야합니다..
        min_after_dequeue=dataset_size)  # 균일한 무작위성을 보장하기 위해 중요합니다.

  return _input_fn

def create_predict_input_fn(features, labels):
  """ 예측을 위한 mnist 데이터를 estimator로 보내는 사용자 input_fn

  Args:
    features: 예측을 위한 특징들.
    labels: 예측 예들을 위한 레이블들.

  Returns:
    훈련시 사용할 특징 및 레이블들을 반환하는 함수
"""
  def _input_fn():
    raw_features = tf.constant(features.values)
    raw_targets = tf.constant(labels.values)
    return tf.train.limit_epochs(raw_features, 1), raw_targets

  return _input_fn

def train_linear_classification_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """MNIST 숫자 데이터 세트로 선형 분류 모델을 훈련합니다.

  이 함수는 훈련에 더하여 시간 경과에 따른 훈련 및 검증 손실의 플롯과 혼동 행렬등의 훈련 진행 정보를 보여줍니다. 

  Args:
    learning_rate: `int`, 사용할 학습율.

    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    batch_size: 0이 아닌 `int`, 배치 크기.
    hidden_units: int 값의 `list`. 각 층 내의 뉴런수.
    training_examples: 훈련을 위한 입력 특징들이 든 `DataFrame`
    training_targets: 훈련을 위한 레이블이 든 `DataFrame`
    validation_examples: 검증을 위한 입력 특징이 든 `DataFrame`
    validation_targets: 검증을 위한 레이블들이 든`DataFrame`

  Returns:
    훈련된 `LinearClassifier` 객체.
  """

  periods = 10
  steps_per_period = steps / periods

  # 입력 함수 만들기
  predict_training_input_fn = create_predict_input_fn(
    training_examples, training_targets)
  predict_validation_input_fn = create_predict_input_fn(
    validation_examples, validation_targets)
  training_input_fn = create_training_input_fn(
    training_examples, training_targets, batch_size)

  # 선형 classifier 만들기
  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
      training_examples)
  classifier = tf.contrib.learn.LinearClassifier(
      feature_columns=feature_columns,
      n_classes=10,
      optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
      gradient_clip_norm=5.0,
      config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
  )

  # 모델을 훈련 시키되 루프 내부에서 수행하여 손실 매트릭을 주기적으로 평가할 수 있습니다.
  print("Training model...")
  print("LogLoss error (on validation data):")
  training_errors = []
  validation_errors = []
  for period in range (0, periods):
    # 이전 상태에서 시작하여 모델을 교육.
    classifier.fit(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # 잠시 멈추고 예측을 계산합니다.
    training_predictions = list(classifier.predict_proba(input_fn=predict_training_input_fn))
    validation_predictions = list(classifier.predict_proba(input_fn=predict_validation_input_fn))
    # 훈련 및 검증 손실 계산.
    training_log_loss = metrics.log_loss(training_targets, training_predictions)
    validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)
    # 주기적으로 현재의 손실을 출력.
    print("  period %02d : %0.2f" % (period, validation_log_loss))
    # 이번 주기의 손실 매트릭을 리스트에 추가.
    training_errors.append(training_log_loss)
    validation_errors.append(validation_log_loss)
  print("Model training finished.")
  # 공간 절약을 위해 이벤트 파일들을 삭제
  _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

  # 최종 예측 계산 (위에서처럼 확률이 아님)
  final_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
  accuracy = metrics.accuracy_score(validation_targets, final_predictions)
  print("Final accuracy (on validation data): %0.2f" % accuracy)

  # 주기에 따른 손실 매트릭 그래프 출력
  plt.clf()
  plt.close()
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.plot(training_errors, label="training")
  plt.plot(validation_errors, label="validation")
  plt.legend()
  plt.show()

  # 혼동 행렬 플롯 출력
  cm = metrics.confusion_matrix(validation_targets, final_predictions)
  # 행을 따라 혼동 행렬을 정규화 (즉, 각 클래스의 샘플수로 정규화)
  cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  plt.figure()
  ax = sns.heatmap(cm_normalized, cmap="bone_r")
  ax.set_aspect(1)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")
  plt.show()

  return classifier

# 5분간 이 형식의 선형 모델로 얼마나 높은 정확도를 얻을 수 있는지 시도해보세요. 이 실습에서는 다른 변경엔 제약을 두고 배치 크기, 학습 속도 및 단계에 대한 하이퍼 매개 변수만 변경하세요.

# 0.9 정도 이상의 정확도를 얻었다면 그만하십시오.

_ = train_linear_classification_model(
    learning_rate=0.02,
    steps=100,
    batch_size=10,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 작업 2 : 선형 분류기를 신경망으로 대체.

# 위의 LinearClassifier를 DNNClassifier로 대체하고 0.95 이상의 정확도를 제공하는 매개 변수 조합을 찾아봅시다.

# 드롭아웃(dropout)과 같은 추가 정규화 방법을 시도해 보고 싶으실겁니다. 이러한 추가적인 정규화 방법은 DNNClassifier 클래스에 대한 주석에 설명되어 있습니다.

#
# 여기에 코드를 작성하세요: 선형 분류기를 신경망으로 대체해보세요.
#

# 좋은 모델을 얻었으면, 아래의 테스트 데이터로 모델을 평가하여 검증 데이터에 오버피팅되지 않았는지 다시 한 번 확인하십시오.

import subprocess
import pathlib

subprocess.run('wget http://datasets.lablup.ai/public/tutorials/mnist_test.csv -O /tmp/mnist_test.csv', shell=True)
filesize = pathlib.Path('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/mnist_test.csv').stat().st_size
print(f'Downloaded mnist_test.csv ({filesize:,} bytes)')

mnist_test_dataframe = pd.read_csv(
  io.open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/mnist_test.csv", "r"),
  sep=",",
  header=None)

test_targets, test_examples = parse_labels_and_features(mnist_test_dataframe)
display(test_examples.describe())

#tensorflow-python3
#
# 여기에 코드를 작성하세요: 테스트 세트로 정확도를 계산하세요.
#

## 작업 3: 첫번째 은닉층의 가중치를 시각화하기

# 잠시 시간을 내어 신경망을 파헤치고 모델의 weights_ 속성에 액세스하여 무엇을 학습했는지 살펴 보겠습니다.

# 우리 모델의 입력층은 28×28 픽셀 입력 이미지에 해당하는 784 가중치를 가집니다. 첫 번째 은닉층은 784×N 가중치를 가지며, 여기서 N은 해당 층의 노드 수입니다. N 1×784의 각 가중치 배열을 28×28 크기의 N 배열로 재구성하여 해당 가중치를 28× 28 이미지로 되돌릴 수 있습니다.

# 다음 셀을 실행하여 가중치를 그려봅시다. 이 셀은 이미 훈련된 DNNClassifier "classifier" 를 필요로합니다.

weights0 = classifier.weights_[0]

print("weights0 shape:", weights0.shape)

num_nodes = weights0.shape[1]
num_rows = int(math.ceil(num_nodes / 10.0))
fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
for coef, ax in zip(weights0.T, axes.ravel()):
    # 계수의 가중치를 1x784 에서 28x28로 모양을 변경합니다.
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.pink)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

# 신경망의 첫번째 은닉층은 꽤 낮은 수준의 특징을 모델링해야 하므로, 가중치를 시각화하면 일부 희미한 얼룩 또는 숫자의 일부만 표시됩니다. 본질적으로는 노이즈인 일부 뉴런도 보일 수 있습니다. 그런 뉴런은 수렴하지 않았거나 상위 계층에서 무시되고 있습니다.

# 서로 다른 반복 횟수로 훈련을 실시하고 횟수에 따른 효과를 확인하는 것도 흥미로울 것입니다.

# 분류기를 10, 100 및 1000 단계로 각각 훈련시키십시오. 그런 다음 이 시각화를 다시 실행하십시오.

# 서로 다른 수렴 수준에서 시각적으로 어떤 차이가 있습니까?