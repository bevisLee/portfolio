### 로지스틱 회귀 (Logistic regression) 기초

## 학습 목표

 # 이진 분류 모델로 중앙 집 값 estimator (이전 연습에서)를 다시 만들어봅니다.
 # 이진 분류 문제에 대한 로지스틱 회귀와 선형 회귀의 효율성을 비교합니다.

# 이전 연습에서와 마찬가지로 캘리포니아 주택 데이터 세트로 작업하지만 이번에는 도시 블록이 비용이 많이 드는 도시 블록인지 여부를 예측하여 이진 분류 문제로 바꿔 놓을 것입니다. 일단 기본 특징(feature)들로 기능으로 되돌리고 시작합시다.

## 문제를 이진 분류로 구성하기

# 데이터 세트의 목표는 지속적으로 가치있는 특징인 median_house_value입니다. 이 연속값에 임계값을 적용하여 부울 레이블을 만들 수 있습니다.

# 도시 블록을 설명하는 특징이 주어지면, 그것이 고비용 도시 블록인지 예측하기를 원합니다. 훈련 및 평가 데이터를 준비하기 위해 집값의 중앙값 (약 265000)에 대해 75% 분류 임계값을 정의합니다. 임계값 위의 모든 집값은 1로 레이블되고 다른 모든 값은 0으로 레이블됩니다.

# 아래의 셀을 실행하여 데이터를 불러오고 입력 특징 및 목표를 준비합니다.

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

# 아래의 코드가 이전 연습과 약간 다른 점에 유의하십시오. median_house_value를 대상으로 사용하는 대신 median_house_value_is_high 라는 새 이진 목표를 만듭니다.

def preprocess_features(california_housing_dataframe):
  """캘리포니아 주택 데이터 셋의 입력 특징을 준비한다.

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
    모델에서 사용할 특징들이 담긴 DataFrame.
  """
  output_targets = pd.DataFrame()
  # `median_house_value`의 값이 임계점(threshold) 보다 큰 값인지 여부
  output_targets["median_house_value_is_high"] = (
    california_housing_dataframe["median_house_value"] > 265000).astype(float)
  return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
display(training_examples.describe())

training_targets = preprocess_targets(california_housing_dataframe.head(12000))
display(training_targets.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
display(validation_examples.describe())

validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
display(validation_targets.describe())

## 선형 회귀 분석은 어떻게 이루어집니까?

# 로지스틱 회귀가 효과적인 이유를 보려면 선형 회귀를 사용하는 간소한 모델을 먼저 훈련 시키십시오. 이 모델은 집합 {0, 1}에있는 값을 가진 레이블을 사용하고 가능한 한 0 또는 1에 가까운 연속 값을 예측하려고 시도합니다. 또한 출력을 확률로 해석하기를 원합니다. 출력이 범위 (0, 1) 내에 있으면 이상적입니다. 그런 다음 레이블을 결정하기 위해 임계값 0.5를 적용합니다.

# 아래의 셀을 실행하여 LinearRegressor를 사용하여 선형 회귀 모델을 학습하십시오.

def train_linear_regressor_model(
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

  # 입력 함수 만들기
  training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets["median_house_value_is_high"],
     num_epochs=None, batch_size=batch_size)
  predict_training_input_fn = learn_io.pandas_input_fn(
     x=training_examples, y=training_targets["median_house_value_is_high"],
     num_epochs=1, shuffle=False)
  predict_validation_input_fn = learn_io.pandas_input_fn(
      x=validation_examples, y=validation_targets["median_house_value_is_high"],
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
        steps=steps_per_period
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
  plt.tight_layout(True)
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()
  plt.show()
  return linear_regressor

linear_regressor = train_linear_regressor_model(
    learning_rate=0.000001,
    steps=200,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 작업 1 : 이러한 예측에 대해 LogLoss를 계산할 수 있을까요?

# 예측을 검토하여 LogLoss 계산에 사용할 수 있는지 여부를 결정하십시오.

# LinearRegressor는 L2 손실을 사용하는데, 이는 출력을 확률로 해석할 때 오분류에 불이익을 주는 경우에는 잘 동작하지 않습니다. 예를 들어, 부정적인 예가 긍정적으로 (잘못) 분류될 때 0.9의 확률 또는 0.9999의 확률로 분류되었는지의 여부에 따라 두 경우에 큰 차이가 있어야 하지만 L2 손실은 이러한 경우를 크게 구분하지 않습니다.

# 대조적으로, LogLoss는 이러한 "신뢰 오류"를 훨씬 더 많이 처벌합니다. LogLoss는 다음과 같이 정의됩니다.

# Log Loss=∑(x,y)−ylog(ypred)−(1−y)log(1−ypred)Log Loss=∑(x,y)−ylog⁡(ypred)−(1−y)log⁡(1−ypred)

# 하지만 먼저 예측 값을 얻어야합니다. LinearRegressor.predict를 사용해 얻을 수 있습니다.

# 예측과 목표로 LogLoss를 계산할 수 있을까요?


## 작업 2 : 로지스틱 회귀 모델을 훈련하고 유효성 검사 집합에서 LogLoss를 계산합니다.

# 로지스틱 회귀 모델을 교육하고 유효성 검사 세트에서 LogLoss를 계산합니다.

# 로지스틱 회귀를 사용하려면 LinearRegressor 대신 LinearClassifier를 사용하기 만하면됩니다.

# Sklearn의 log_loss 함수는 LogLoss를 계산할 때 유용합니다.

# LinearClassifier.predict는 훈련된 모델로부터 예측을 얻는 데 유용합니다. 하지만 여기서는 부울 예측 클래스가 아닌 실제 값 확률이 필요합니다! 다행히 LinearClassifier.predict_proba 가 바로 그 역할을 합니다. 그러나 predict_proba는 확률에 대해 두 개의 열을 반환합니다. 하나는 0 값이고 다른 하나는 값 1입니다. 값 1에 대한 확률 만 있으면 됩니다.

def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """하나의 특징에 대한 선형 회귀 모델을 훈련합니다.

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
    훈련 데이터로 훈련한 `LinearClassifier` 객체.
  """

  #
  # 여기에 코드를 작성하세요: 모델을 훈련하고 검증 세트로 LogLoss를 계산하기 위해 LinearClassifier를 사용하세요.
  #

  pass

linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

## 작업 3 : 정확도를 계산하고 유효성 검사 세트에서 ROC 곡선 AUC를 그립니다.

# 분류에 유용한 몇 가지 측정법은 모델 정확도, ROC 곡선 및 ROC 곡선 (AUC) 아래의 면적입니다. 이러한 측정 항목을 살펴 보겠습니다.

# LinearClassifier.evaluate는 정확도 및 AUC와 같은 유용한 통계를 계산합니다.

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])

# Linear Classifier.predict_proba 및 Sklearn roc_curve에서 계산 한 것과 같은 클래스 확률을 사용하여 ROC 곡선을 그리는 데 필요한 실제 양성률 및 위양성 비율을 얻을 수 있습니다.

validation_probabilities = np.array(list(linear_classifier.predict_proba(
    input_fn=predict_validation_input_fn)))

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_probabilities[:, 1])
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)

# AUC를 향상시키기 위해 작업 2에서 훈련된 모델의 학습 설정을 조정할 수 있는지 확인하십시오.

# 때로는 특정 메트릭은 다른 메트릭을 손상시키면서 개선되기 때문에, 좋은 절충안을 제공하는 설정을 찾아야 합니다.

# 동시에 모든 메트릭이 향상되는지 확인하십시오.

linear_classifier = train_linear_classifier_model(
    learning_rate=0.000005,
    steps=500,
    batch_size=20,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])