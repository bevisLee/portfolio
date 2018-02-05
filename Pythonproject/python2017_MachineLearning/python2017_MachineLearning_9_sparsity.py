### 띄엄띄엄함(sparsity) 및 L1 정규화

## 학습 목표

# 모델의 크기를 계산해봅니다.
# 띄엄띄엄함(sparsity)을 증가시켜 모델의 크기를 줄이기 위해 L1 정규화를 적용해봅니다.
# 복잡성을 줄이는 한 가지 방법은 가중치를 정확하게 0으로 만드는 정규화 함수를 사용하는 것입니다. 회귀와 같은 선형 모델의 경우, 0 가중치는 해당 특징을 전혀 사용하지 않는 것과 같습니다. 오버 피팅 (overfitting)을 피하는 것 외에도 결과 모델이 더 효율적입니다.

# L1 정규화는 띄엄띄엄함(희소성, sparsity)을 높이는 좋은 방법입니다.

# 아래의 셀을 실행하여 데이터를 불러오고 특징 정의를 해 봅시다.

import math

from sorna.display import display
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
  """캘리포니아 주택 데이터 세트로부터 입력 특징을 준비한다.

  Args:
    california_housing_dataframe: 캘리포니아 주택 데이터 셋을 포함한 Pandas DataFrame
  Returns:
    합성 특징을 포함한, 모델에서 사용할 특징들이 담긴 DataFrame.
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
  # 합성 특징을 만든다.
  processed_features["rooms_per_person"] = (
    california_housing_dataframe["total_rooms"] /
    california_housing_dataframe["population"])
  return processed_features

def preprocess_targets(california_housing_dataframe):
  """캘리포니아 주택 데이터 세트로부터 목표 특징 (레이블)을 준비한다.

  Args:
    california_housing_dataframe: 캘리포니아 주택 데이터 세트을 포함한 Pandas DataFrame
  Returns:
    모델에서 사용할 목표 특징들이 담긴 DataFrame.
  """
  output_targets = pd.DataFrame()
  # `median_house_value`의 값이 임계점(threshold) 보다 큰 값인지 여부
  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)
  return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

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
    `(features, target_tensor)` `tuple`:
      features: 입력 특징으로 사용할 특징 `dict`.
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

def get_quantile_based_buckets(feature_values, num_buckets):
  """quantile 기반의 버킷을 반환합니다.

  Returns:
    quantiles: 각 그룹의 경계 값을 반환하는 `float`

  Example:
    >>> get_quantile_based_buckets(training_examples["households"], 5)
      [236.0, 324.0, 411.0, 522.0, 721.0]
  """
  quantiles = feature_values.quantile(
    [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
  return [quantiles[q] for q in quantiles.keys()]

bucketized_households = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("households"),
  boundaries=get_quantile_based_buckets(training_examples["households"], 10))
bucketized_longitude = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("longitude"),
  boundaries=get_quantile_based_buckets(training_examples["longitude"], 50))
bucketized_latitude = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("latitude"),
  boundaries=get_quantile_based_buckets(training_examples["latitude"], 50))
bucketized_housing_median_age = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("housing_median_age"),
  boundaries=get_quantile_based_buckets(
    training_examples["housing_median_age"], 10))
bucketized_total_rooms = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("total_rooms"),
  boundaries=get_quantile_based_buckets(training_examples["total_rooms"], 10))
bucketized_total_bedrooms = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("total_bedrooms"),
  boundaries=get_quantile_based_buckets(training_examples["total_bedrooms"], 10))
bucketized_population = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("population"),
  boundaries=get_quantile_based_buckets(training_examples["population"], 10))
bucketized_median_income = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("median_income"),
  boundaries=get_quantile_based_buckets(training_examples["median_income"], 10))
bucketized_rooms_per_person = tf.contrib.layers.bucketized_column(
  tf.contrib.layers.real_valued_column("rooms_per_person"),
  boundaries=get_quantile_based_buckets(
    training_examples["rooms_per_person"], 10))

long_x_lat = tf.contrib.layers.crossed_column(
  set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)

feature_columns = set([
  long_x_lat,
  bucketized_longitude,
  bucketized_latitude,
  bucketized_housing_median_age,
  bucketized_total_rooms,
  bucketized_total_bedrooms,
  bucketized_population,
  bucketized_households,
  bucketized_median_income,
  bucketized_rooms_per_person])

## 모델 크기 계산

# 모델 크기를 계산하기 위해서는 0이 아닌 매개 변수의 수를 단순히 세면 됩니다. 아래의 도우미 기능이 제공됩니다. 이 함수는 Estimators API에 대한 자세한 지식을 사용합니다 - API가 작동하는 방식에 대해 이해하지 않아도 되니 걱정하지 않아도 됩니다.

def model_size(estimator):
  variables = estimator.get_variable_names()
  size = 0
  for variable in variables:
    if not any(x in variable 
               for x in ['global_step',
                         'centered_bias_weight',
                         'bias_weight',
                         'Ftrl']
              ):
      size += np.count_nonzero(estimator.get_variable_value(variable))
  return size

## 모델 크기 줄이기

# 당신의 팀은 도시 블록 ('median_income', 'avg_rooms', 'households'... 등)의 인구 통계를 감지 하고 주어진 도시 블록이 고비용 도시 블록인지 여부를 알려줄 수 있는 똑똑한 반지 인 SmartRing에 매우 정확한 로지스틱 회귀 모델을 구축할 필요가 있습니다.

# SmartRing이 작기 때문에 엔지니어링 팀은 매개 변수가 600 개 이하 인 모델만 처리 할 수 있다고 결정했습니다. 반면 제품 관리 팀은 출하 테스트에서 LogLoss가 0.35 미만이 아닌 이상 모델을 출시 할 수 없다고 결정했습니다.

# 비밀 무기 (L1 정규화)를 사용하여 크기와 정확도 제약 조건을 모두 만족하도록 모델을 조정할 수 있을까요?

## 작업 1 : 좋은 정규화 계수 찾기

# 모델 크기가 600 미만이고 유효성 검사 집합에서 로그 손실이 0.35 미만인 두 제약 조건을 모두 만족하는 L1 정규화 강도 매개 변수를 찾습니다.

# 아래 코드는 시작하는 데 도움이 됩니다. 모델에 정규화를 적용하는 데는 여러 가지 방법이 있습니다. 여기서는 표준 경사 하강보다 L1 정규화에서 더 나은 결과를 제공하도록 설계된 FtrlOptimizer를 사용하기로 결정했습니다.

# 모델이 전체 데이터 세트에 대해 교육을 수행하므로 정상보다 느리게 실행될 것을 염두에 두세요.

def train_linear_classifier_model(
    learning_rate,
    regularization_strength,
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
    regularization_strength: L1 정규화의 세기를 나타내는 `float`.
      `0.0`은 정규화를 하지 않는 것을 의미합니다.
    steps: 0이 아닌 `int`, 총 훈련 단계 수. 
      훈련 단계는 단일 배치를 사용하는 전진 및 역진 통과(forward/backward pass)로 구성됩니다. 
    feature_columns: 사용할 입력 특징 열을 지정하는 `set`.
    training_examples: 훈련을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    training_targets: 훈련을 위한 타겟으로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`
    validation_examples: 검증을 위한 입력 특징으로 사용할 `california_housing_dataframe`
      내의 하나 또는 여러개의 열이 든 `DataFrame`
    validation_targets: 검증을 위한 타겟으로 사용할 `california_housing_dataframe`
      내의 하나의 열이 든 `DataFrame`

  Returns:
    훈련 데이터로 훈련한 `LinearClassifier` 객체.
  """
  periods = 7
  steps_per_period = steps / periods

  # 선형 회귀 객체 생성.
  linear_classifier = tf.contrib.learn.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=tf.train.FtrlOptimizer(
          learning_rate=learning_rate,
          l1_regularization_strength=regularization_strength),
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
  print("LogLoss (on validation data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # 이전 상태에서 시작하여 모델을 교육.
    linear_classifier.fit(
        input_fn=training_input_function,
        steps=steps_per_period
    )
    # 잠시 멈추고 예측을 계산합니다.
    training_probabilities = np.array(list(linear_classifier.predict_proba(
        input_fn=training_input_function_for_predict)))
    validation_probabilities = np.array(list(linear_classifier.predict_proba(
          input_fn=validation_input_function_for_predict)))
    # 훈련 및 검증 손실 계산.
    training_log_loss = metrics.log_loss(training_targets, training_probabilities[:, 1])
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities[:, 1])
    # 주기적으로 현재의 손실을 출력.
    print("  period %02d : %0.2f" % (period, validation_log_loss))
    # 이번 주기의 손실 매트릭을 리스트에 추가.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")

  # 주기에 따른 손실 매트릭 그래프 출력
  plt.clf()
  plt.close()
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout(True)
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()
  plt.show()

  return linear_classifier

linear_classifier = train_linear_classifier_model(
    learning_rate=0.1,
    regularization_strength=0.0,
    steps=300,
    feature_columns=feature_columns,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
print("Model size:", model_size(linear_classifier))

