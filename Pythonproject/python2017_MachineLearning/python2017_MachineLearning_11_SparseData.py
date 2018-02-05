### 띄엄띄엄한 데이터와 임베딩 기초

## 학습 목표

# 영화 리뷰 문자열 데이터를 띄엄띄엄한 특징 벡터로 변환합니다.
# 띄엄띄엄한 특징 벡터를 사용하여 정서 분석 선형 모델을 구현해봅니다.
# 데이터를 2차원으로 투영하는 임베딩을 사용하여 정서 분석 DNN 모델을 구현해봅니다.
# 임베딩을 시각화하여 단어 사이의 관계에 대해 모델이 배운 것을 확인해봅니다.
# 이 실습에서는 무비 리뷰의 텍스트 데이터 (ACL 2011 IMDB dataset)를 사용하여 띄엄띄엄한 데이터를 탐색하고 임베딩 작업을 합니다. 이 데이터는 이미 tf.Example 형식으로 사전 처리되었습니다.
 # data - http://ai.stanford.edu/~amaas/data/sentiment/

# 우선 훈련 및 테스트 데이터를 다운로드하십시오.

import os
import subprocess
import pathlib

subprocess.run('wget http://datasets.lablup.ai/public/tutorials/test.tfrecord -O /tmp/test.tfrecord', shell=True)
filesize = pathlib.Path('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/test.tfrecord').stat().st_size
print(f'Downloaded test data ({filesize:,} bytes)')
subprocess.run('wget http://datasets.lablup.ai/public/tutorials/train.tfrecord -O /tmp/train.tfrecord', shell=True)
filesize = pathlib.Path('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord').stat().st_size
print(f'Downloaded train data ({filesize:,} bytes)')

# 리뷰가 일반적으로 호의적 (1 점의 라벨) 또는 비호의적 (0 점의 라벨)인지 예측하는 데이터에 대한 정서 분석 모델을 교육합시다.

# 그렇게하기 위해 데이터에서 볼 수있는 각 용어의 목록 인 어휘를 사용하여 문자열 '용어'를 특징 벡터로 변환합니다. 이 연습의 목적을 위해 한정된 용어 집합에 초점을 맞춘 작은 어휘를 만들었습니다. 이 용어의 대부분은 호의적 또는 비호의적을 나타내는 것으로 밝혀졌지만 일부는 재미 있기 때문에 추가되었습니다.

# 어휘의 각 용어는 특징 벡터의 좌표에 매핑됩니다. 예제의 문자열 값 terms를 벡터 형식으로 변환하기 위해 예제 문자열에 어휘 용어가 표시되지 않으면 각 좌표가 0 값을 갖도록 인코딩하고 그렇지 않으면 1 값을 얻도록 인코딩합니다. 어휘에 나타나지 않는 예제의 용어는 버려집니다.

# NOTE - 우리는 물론 더 큰 어휘를 사용할 수 있으며 이를 생성하기위한 특별한 도구가 있습니다. 또한 어휘에 없는 용어를 삭제하는 대신 어휘에 포함되지 않은 용어를 해싱 할 수있는 소수의 OOV (어휘 밖, out-of-vocabulary) 버킷을 도입 할 수 있습니다. 우리는 명시적 어휘를 생성하는 대신 각 용어를 해시하는 feature hashing 접근법을 사용할 수도 있습니다. 이것은 실제로 잘 작동하지만 연습할 때 유용한 해석 가능성을 떨어뜨립니다.

## 작업 1 : 띄엄띄엄한 입력과 명확한 어휘로 선형 모델을 사용하기

# 먼저 54 개의 용어를 사용하여 LinearClassifier 모델을 만들 것입니다. 항상 간단하게 시작합시다!

# sparse_column_with_keys 함수는 문자열-특징-벡터 매핑을 사용하여 특징 열을 생성합니다.

# 코드를 읽은 후에 실행하고 어떻게 돌아가는지 확인하십시오.

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from sorna.display import display
from IPython.display import display

from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)

# 우선, `tf.Examples`로부터 특징을 추출하여 사전(dict)을 만듭니다.
features_to_types_dict = {
    "terms": tf.VarLenFeature(dtype=tf.string),
    "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)}

# 주어진 파일 패턴으로부터 tf.Examples 를 추출하는 input_fn 을 만들고, 특징 및 목적으로 나눕니다.
def _input_fn(input_file_pattern):
  features = tf.contrib.learn.io.read_batch_features(
    file_pattern=input_file_pattern,
    batch_size=25,
    features=features_to_types_dict,
    reader=tf.TFRecordReader)
  targets = features.pop("labels")
  return features, targets

# 모델 단어목록을 구성하는 우리가 선택한 54개의 의미 용어들
informative_terms = [ "bad", "great", "best", "worst", "fun", "beautiful",
                      "excellent", "poor", "boring", "awful", "terrible",
                      "definitely", "perfect", "liked", "worse", "waste",
                      "entertaining", "loved", "unfortunately", "amazing",
                      "enjoyed", "favorite", "horrible", "brilliant", "highly",
                      "simple", "annoying", "today", "hilarious", "enjoyable",
                      "dull", "fantastic", "poorly", "fails", "disappointing",
                      "disappointment", "not", "him", "her", "good", "time",
                       "?", ".", "!", "movie", "film", "action", "comedy",
                       "drama", "family", "man", "woman", "boy", "girl" ]

# 의미 용어들을 이용해서 "terms"로부터 특징 열을 만듭니다.
terms_feature_column = \
  tf.contrib.layers.sparse_column_with_keys(column_name="terms",
                                            keys=informative_terms)

feature_columns = [ terms_feature_column ]

classifier = tf.contrib.learn.LinearClassifier(
  feature_columns=feature_columns,
  optimizer=tf.train.AdagradOptimizer(
    learning_rate=0.1),
  gradient_clip_norm=5.0
)

classifier.fit(
  input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord"),
  steps=1000)

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord"),
  steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/test.tfrecord"),
  steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

## 작업 2 : 심층신경망 네트워크 (DNN) 모델 사용

# 위의 모델은 선형 모델입니다. 아주 잘 작동합니다. DNN 모델을 사용하면 더 잘 할 수 있을까요? 해 봅시다.

# LinearClassifier를 DNNClassifier를 바꾸어 보겠습니다. 다음 셀을 실행하십시오. 어떤 종류의 오류가 발생합니까?

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from sorna.display import display
from IPython.display import display

from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)

# 우선, `tf.Examples`로부터 특징을 추출하여 사전(dict)을 만듭니다.
features_to_types_dict = {
    "terms": tf.VarLenFeature(dtype=tf.string),
    "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)}

# 주어진 파일 패턴으로부터 tf.Examples 를 추출하는 input_fn 을 만들고, 특징 및 목적으로 나눕니다.
def _input_fn(input_file_pattern):
  features = tf.contrib.learn.io.read_batch_features(
    file_pattern=input_file_pattern,
    batch_size=25,
    features=features_to_types_dict,
    reader=tf.TFRecordReader)
  targets = features.pop("labels")
  return features, targets

informative_terms = [ "bad", "great", "best", "worst", "fun", "beautiful",
                      "excellent", "poor", "boring", "awful", "terrible",
                      "definitely", "perfect", "liked", "worse", "waste",
                      "entertaining", "loved", "unfortunately", "amazing",
                      "enjoyed", "favorite", "horrible", "brilliant", "highly",
                      "simple", "annoying", "today", "hilarious", "enjoyable",
                      "dull", "fantastic", "poorly", "fails", "disappointing",
                      "disappointment", "not", "him", "her", "good", "time",
                       "?", ".", "!", "movie", "film", "action", "comedy",
                       "drama", "family", "man", "woman", "boy", "girl" ]

# 특징 해싱을 이용해서 "terms"로부터 특징 열을 만듭니다.
terms_feature_column = tf.contrib.layers.sparse_column_with_keys(column_name="terms",
                                                                 keys=informative_terms)

feature_columns = [ terms_feature_column ]

########################### 이 부분을 변경했습니다 ##################################
classifier = tf.contrib.learn.DNNClassifier(                                  #
  feature_columns=feature_columns,                                            #
  hidden_units=[20,20],                                                       #
  optimizer=tf.train.AdagradOptimizer(                                        #
    learning_rate=0.1),                                                       #
  gradient_clip_norm=5.0                                                      #
)                                                                             #
###############################################################################

try:
  classifier.fit(
    input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord"),
    steps=1000)

  evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord"),
    steps=1)
  print("Training set metrics:")
  for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
  print("---")

  evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/test.tfrecord"),
    steps=1)

  print("Test set metrics:")
  for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
  print("---")
except ValueError as err:
  print(err)

## 작업 3 : DNN 모델에 임베딩 사용

# 띄엄띄엄한 특징 열 (예 :_SparseColumnKeys)을 사용하여 DNNClassifier를 교육하면 오류가 발생합니다. DNNClassifier에 드문드문 한 입력 데이터를 전달하는 방법이 필요합니다. 작업 2에 대한 오류 출력에서 보았듯, 두 가지 옵션이 있습니다 : 임베딩 열 (embedding_column) 또는 one-hot 열(one_hot_column) 입니다.

# 이 작업에서는 임베딩 열을 사용하여 DNN 모델을 구현합니다. 임베딩 열은 띄엄띄엄한 데이터를 입력으로 사용하고 더 낮은 차원의 고밀도 벡터를 출력으로 반환합니다.

# NOTE - embedding_column은 일반적으로 띄엄띄엄한 데이터에서 모델을 학습하는 데 사용하는 계산적으로 가장 효율적인 옵션입니다. 이 연습의 마지막 부분에 있는 선택 항목에서, embedding_column 과 one_hot_column을 사용하는 것 사이의 구현상의 차이점과 함께 다른 하나를 선택하는 것의 절충점에 대해 더 자세히 논의 할 것입니다.

# 아래 코드에서 다음을 수행하십시오.

# 데이터를 2차원으로 투영하는embedding_column을 사용하여 모델의 특징 컬럼을 정의하십시오 (embedding_column에 대한 함수 시그니처에 대한 자세한 내용은 TF docs 을 확인하세요).
# 다음의 스펙을 가진DNNClassifier를 정의하십시오 :
  # * 각 20 유닛의 두개의 히든 레이어
  # * 0.1의 학습 속도로 Adagrad 최적화
  # * 5.0의gradient_clip_norm

# NOTE - 실제로는 50또는 100차원처럼 2보다 큰 차원에 투영할 수 있습니다. 하지만 지금은 2차원이 시각화하기 쉽습니다.

# 힌트: 특징 열을 정의하는데 사용할 수 있는 코드 예입니다. (이 예제로도 잘 돌아갑니다)
terms_embedding_column = tf.contrib.layers.embedding_column(terms_feature_column, dimension=2)
feature_columns = [ terms_embedding_column ]

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# from sorna.display import display
from IPython.display import display

from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)

# 우선, `tf.Examples`로부터 특징을 추출하여 사전(dict)을 준비합니다.
features_to_types_dict = {
    "terms": tf.VarLenFeature(dtype=tf.string),
    "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)}

# 주어진 파일 패턴으로부터 tf.Examples 를 추출하는 input_fn 을 만들고, 특징 및 목적으로 나눕니다.
def _input_fn(input_file_pattern):
  features = tf.contrib.learn.io.read_batch_features(
    file_pattern=input_file_pattern,
    batch_size=25,
    features=features_to_types_dict,
    reader=tf.TFRecordReader)
  targets = features.pop("labels")
  return features, targets


informative_terms = [ "bad", "great", "best", "worst", "fun", "beautiful",
                      "excellent", "poor", "boring", "awful", "terrible",
                      "definitely", "perfect", "liked", "worse", "waste",
                      "entertaining", "loved", "unfortunately", "amazing",
                      "enjoyed", "favorite", "horrible", "brilliant", "highly",
                      "simple", "annoying", "today", "hilarious", "enjoyable",
                      "dull", "fantastic", "poorly", "fails", "disappointing",
                      "disappointment", "not", "him", "her", "good", "time",
                       "?", ".", "!", "movie", "film", "action", "comedy",
                       "drama", "family", "man", "woman", "boy", "girl" ]

# 특징 해싱을 이용해서 "terms"로부터 특징 열을 만듭니다.
terms_feature_column = tf.contrib.layers.sparse_column_with_keys(column_name="terms",
                                                                 keys=informative_terms)

########################## 수정할 코드를 입력할 부분입니다##############################
terms_embedding_column = # 임베딩 열을 정의하세요
feature_columns = # 특징 열을 정의하세요

classifier = # DNNClassifier 를 정의하세요
################################################################################

classifier.fit(
  input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord"),
  steps=1000)

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/train.tfrecord"),
  steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/test.tfrecord"),
  steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

## 작업 4 : 실제로 거기에 임베딩이 있다고 확신하기

# 위의 모델은embedding_column을 사용했고 작동하는 것으로 보였습니다. 그러나 이것은 내부적으로 어떤 일이 벌어지고 있는지에 대해서는 별로 알려주지 않습니다. 모델이 실제로 내부에 임베딩을 사용하고 있는지 확인할 수 있습니까?

# 모델의 텐서를 살펴 봅시다.

print(classifier.get_variable_names())

# 거기에 embedding layer가 있음을 볼 수 있습니다 :'dnn/input_from_feature_columns/terms_embedding/...'. (여기서 흥미로운 점은 히든 레이어와 마찬가지로 모델의 나머지 부분과 함께 이 레이어를 훈련 할 수 있다는 것입니다.)

# 임베딩 레이어가 올바른 모양입니까? 확인해 보기 위해 다음 코드를 실행하십시오.

# NOTE - 우리의 경우 embedding은 54 차원 벡터를 2차원으로 투영하는 행렬임을 기억하십시오.

print(classifier.get_variable_value('dnn/input_from_feature_columns/terms_embedding/weights').shape)

# 다양한 레이어와 모양을 보고 모든 것이 예상대로 연결되었는지 시간을 들여 수동으로 확인해보세요.

## 작업 5 : 임베딩 검사

# 이제 실제 임베딩 공간을 살펴보고 용어들이 어디에서 끝나는 지 살펴 보겠습니다. 다음을 수행하십시오.

# 다음 코드를 실행하여 우리가 작업 3에서 훈련시킨 임베딩을 확인하십시오. 생각했던 대로 되어 있습니까?
# 작업 3 코드를 재실행하여 모델을 다시 훈련시킨 다음 아래에서 임베딩 시각화를 다시 실행하십시오. 무엇이 같습니까? 무엇이 달라졌습니까?
# 마지막으로 10 단계만 실행하여 모델을 다시 훈련하십시오 (끔찍한 모델이 나올겁니다). 아래에서 임베딩 시각화를 다시 실행하십시오. 무엇이 보입니까? 이유는 무엇입니까?

import numpy as np
import matplotlib.pyplot as plt

embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/terms_embedding/weights')

for term_index in range(len(informative_terms)):
  # 단어들의 one-hot 인코딩을 만듭니다. 각 단어에 대응하는 하나의 축만 1이고 나머지는 0으로 채워집니다.
  term_vector = np.zeros(len(informative_terms))
  term_vector[term_index] = 1
  # 이제 임베딩 공간에 one-hot 벡터를 투영합니다.
  embedding_xy = np.matmul(term_vector, embedding_matrix)
  plt.text(embedding_xy[0],
           embedding_xy[1],
           informative_terms[term_index])

# 살짝 설정을 고쳐서 플롯을 멋지게 만들어봅시다.
plt.rcParams["figure.figsize"] = (12, 12)
plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
plt.show() 

## 작업 6 : 모델의 성능 향상

# 성능을 향상시키기 위해 모델을 수정할 수 있는지 확인하십시오. 몇 가지를 시도해 볼 수 있습니다 :

# 하이퍼 매개 변수 변경 또는 Adam과 같은 다른 옵티마이저를 사용하는 것 (이 전략에 따라 하나 또는 두 개의 정확도 백분율 만 개선할 수 있을겁니다).
# informative_terms에 추가 용어 추가 https://storage.googleapis.com/advanced-solutions-lab/mlcc/sparse_data_embedding/terms.txt 에 이 데이터 세트에 대한 모든 30716 용어가 포함된 완전한 어휘 파일이 있습니다. 이 어휘 파일에서 추가 용어를 선택하거나 sparse_column_with_vocabulary_file 특징 열을 이용해 모두 사용할 수 있습니다.

import os
import subprocess
import pathlib

subprocess.run('wget http://datasets.lablup.ai/public/tutorials/sparse_data_embedding_terms.txt -O /tmp/terms.txt', shell=True)
filesize = pathlib.Path('/tmp/terms.txt').stat().st_size
print(f'Downloaded term data ({filesize:,} bytes)')

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sorna.display import display
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)

# 우선, `tf.Examples`로부터 특징을 추출하여 사전(dict)을 준비합니다.
features_to_types_dict = {
    "terms": tf.VarLenFeature(dtype=tf.string),
    "labels": tf.FixedLenFeature(shape=[1], dtype=tf.float32)}

# 주어진 파일 패턴으로부터 tf.Examples 를 추출하는 input_fn 을 만들고, 특징 및 목적으로 나눕니다.
def _input_fn(input_file_pattern):
  features = tf.contrib.learn.io.read_batch_features(
    file_pattern=input_file_pattern,
    batch_size=100,
    features=features_to_types_dict,
    reader=tf.TFRecordReader)
  targets = features.pop("labels")
  return features, targets

# 전체 단어목록 파일을 사용하여 "terms"로부터 특징 열을 만듭니다.
informative_terms = None
with open("/tmp/terms.txt", 'r') as f:
  # 중복을 제거하기 위해 set 타입으로 변환합니다.
  informative_terms = list(set(f.read().split()))
terms_feature_column = tf.contrib.layers.sparse_column_with_keys(column_name="terms",
                                                                 keys=informative_terms)

terms_embedding_column = tf.contrib.layers.embedding_column(terms_feature_column, dimension=2)
feature_columns = [ terms_embedding_column ]

classifier = tf.contrib.learn.DNNClassifier(
  feature_columns=feature_columns,
  hidden_units=[10, 10],
  optimizer=tf.train.AdamOptimizer(
    learning_rate=0.001),
  gradient_clip_norm=1.0
)

classifier.fit(
  input_fn=lambda: _input_fn("/tmp/train.tfrecord"),
  steps=1000)

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn("/tmp/train.tfrecord"),
  steps=1000)
print("Training set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

evaluation_metrics = classifier.evaluate(
  input_fn=lambda: _input_fn("/tmp/test.tfrecord"),
  steps=1000)

print("Test set metrics:")
for m in evaluation_metrics:
  print(m, evaluation_metrics[m])
print("---")

## 마치며

# 우리는 임베딩으로 원래의 선형 모델보다 우수한 DNN 솔루션을 얻을 수 있었지만 선형 모델도 꽤 좋았고 훈련하기가 훨씬 빨랐습니다. 선형 모델은 보정 할 수 있는 매개 변수가 거의 없거나 역전파(backprop)할 계층이 없기 때문에 더 빨리 훈련합니다.

# 일부 응용 프로그램에서는 선형 모델의 속도가 게임 체인저일 수도 있고 선형 모델이 품질면에서 완벽 할 수도 있습니다. 다른 영역에서는 DNN이 제공하는 추가 모델의 복잡성과 용량이 더 중요할 수 있습니다. 모델 아키텍처를 정의 할 때는 문제를 충분히 분석하여 어떤 영역에 있는지를 아는 것이 중요하다는 것을 기억하세요.

## 선택 토론 : embedding_column과 one_hot_column 사이의 장단점

# 개념적으로LinearClassifier 나DNNClassifier를 훈련할 때, 띄엄띄엄한 데이터 열을 사용하는 데 필요한 어댑터가 있습니다. TF는embedding_column 또는one_hot_column 옵션을 제공합니다.

# LinearClassifier를 (작업 1에서와 같이) 훈련 시키면 내부적으로 'embedding_column'이 사용됩니다. 작업 2에서 볼 수 있듯이,DNNClassifier를 훈련 할 때는 embedding_column 또는one_hot_column을 명시적으로 선택해야합니다. 이 절에서는 두 가지의 차이점과 간단한 예제를 살펴보면서 두 가지를 사용하는 것의 절충에 대해 설명합니다.

# "great","beautiful","excellent"가 포함된 띄엄띄엄한 데이터가 있다고 가정합시다. 여기서 사용하는 어휘 크기는 V=54V=54 이기 때문에 첫번째 계층의 각 단위 (뉴런)는 54 개의 가중치를 가집니다. 띄엄띄엄한 입력에서 ss 는 용어의 수를 나타냅니다. 따라서 이 예제의 띄엄띄엄한 데이터의 경우 s=3s=3 입니다. 가능한 값이 VV 인 입력 레이어의 경우 dd 단위를 갖는 숨겨진 레이어는 (1 timesV)∗(V timesd)(1 timesV)∗(V timesd) 의 벡터 행렬 곱셈을 수행해야합니다. 이것의 계산 비용은 O(V∗d)O(V∗d)입니다. 이 비용은 숨겨진 레이어의 가중치 수 및 ss 와는 무관합니다.

# one_hot_column을 사용하여 입력이 one-hot-encoded 된 경우 (길이가 VV 인 부울 벡터, 존재하는 용어는 1, 나머지는 0), 이것은 많은 0을 곱하고 더하는 것을 의미합니다.

# 크기 dd의 embedding_column을 사용하여 똑같은 결과를 얻었을 때, 우리는 "great","beautiful","excellent"이라는 예제 입력에 나타난 세 가지 특징에 해당하는 임베딩을 찾아서 추가합니다: (1×d)+(1×d)+(1×d)(1×d)+(1×d)+(1×d). 없는 특징에 대한 가중치는 벡터-행렬 곱셈에서 0을 곱하기 때문에 결과에 영향을 주지 않습니다. 현재 존재하는 특징에 대한 가중치는 벡터-행렬 곱셈에서 1을 곱합니다. 따라서 임베딩 룩업을 통해 얻은 가중치를 더하면 벡터-행렬 곱셈과 동일한 결과가 나타납니다.

# 임베딩을 사용할 때 임베딩 룩업 (embedding lookup)을 계산하는 것은 O(s∗d)O(s∗d) 계산이므로, ss가 있는 듬성듬성한 데이터 (ss 는 VV 보다 훨씬 작습니다) 의 one_hot_column에 대한 O(V∗d)O(V∗d) 비용보다 계산상 훨씬 효율적입니다. (이러한 임베딩은 학습된 것임을 기억하십시오. 어떤 주어진 훈련 반복에서 그것은 룩업되고 있는 현재 가중치입니다.)

# 작업 3에서 보았듯,DNNClassifier를 훈련 할 때embedding_column을 사용함으로써 모델은 특징에 대한 저차원 표현을 학습합니다. 여기서 내적(dot product)은 원하는 작업에 맞는 유사도를 정의합니다. 이 예에서 영화 리뷰의 맥락에서 유사하게 사용되는 용어 (예 :"great" 및 "excellent")는 임베딩 공간에서 서로 가깝게 위치합니다 (즉, 큰 내적값을 갖습니다). 유사하지 않은 용어 (예 :"great"과"bad")는 임베딩 공간에서 서로 멀어질 것입니다 (즉, 작은 내적값을 가질것입니다).

