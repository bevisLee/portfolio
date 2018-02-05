### Pandas 기본

## 학습 목표
# pandas 라이브러리의 DataFrame 및 Series 데이터 구조에 대한 소개.
# DataFrame 및 Series 내에서 데이터 액세스 및 조작을 해 봅니다.
# pandas DataFrame으로 CSV 데이터 가져 오기를 해 봅니다.
# DataFrame을 다시 색인화하여 데이터를 임의로 처리하는 법을 익힙니다.

## pandas
# pandas는 열 기반 데이터 분석 API입니다. 입력 데이터를 처리하고 분석하는 데 유용한 도구이며 많은 ML 프레임 워크는 pandas 데이터 구조를 입력으로 지원합니다. API에 대한 포괄적인 소개가 많은 페이지에 걸쳐 있지만 핵심 개념은 매우 간단하며 아래에서 설명 할 것입니다. 보다 완벽한 참조를 위해 pandas docs 사이트에는 광범위한 문서와 많은 자습서가 포함되어 있습니다.

## 기본 컨셉
# 아래 명령은 pandas API를 불러오고 API의 버전을 출력합니다.

import pandas as pd
# from sorna.display import display
from IPython.display import display

print(pd.__version__)

# pandas의 기본 데이터 구조는 두 가지 클래스로 구현됩니다.

 # DataFrame: 관계형 데이터 테이블로 상상할 수있는 행 및 명명 된 열
 # Series: 단일 열입니다. DataFrame은 하나 이상의 Series와 각 Series의 이름을 포함합니다.
# 데이터 프레임은 데이터 조작을 위해 일반적으로 사용되는 추상화입니다. Spark 및 R에도 비슷한 구현이 존재합니다.

# Series를 만드는 한 가지 방법은 Series 객체를 생성하는 것입니다. 예를 들면:
pd.Series(['San Francisco', 'San Jose', 'Sacramento'])

# dict 매핑 문자열 열 이름을 각각의 Series에 전달하여 DataFrame 객체를 만들 수 있습니다. 시리즈 길이가 일치하지 않으면 누락 된 값은 특별한 NA/NaN 값으로 채워집니다.

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

display(pd.DataFrame({ 'City name': city_names, 'Population': population }))

# 하지만 대부분의 경우 전체 파일을 DataFrame에로드합니다. 다음 예제는 캘리포니아 주거 데이터가 있는 파일을 로드합니다. 아래의 셀을 실행하여 데이터를 로드하고 feature 정의를 작성하십시오.

# from sorna.display import display
from IPython.display import display

california_housing_dataframe = pd.read_csv("http://datasets.lablup.ai/public/tutorials/california_housing_train.csv", sep=",")
display(california_housing_dataframe.describe())

# 위 예제는 DataFrame에 대한 흥미로운 통계를 보여주기 위해 DataFrame.describe를 사용했습니다. 또 다른 유용한 기능은 DataFrame.head입니다. 처음 몇 레코드를 볼 수 있습니다.
display(california_housing_dataframe.head())

# pandas의 또 다른 강력한 기능은 그래프입니다. 예를 들어, DataFrame.hist를 사용하면 열의 값 분포를 빠르게 연구 할 수 있습니다.
import matplotlib.pyplot as plt
california_housing_dataframe.hist('housing_median_age')
plt.show()

## 데이터 액서스
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(type(cities['City name']))
print(cities['City name'])

print(type(cities['City name'][1]))
print(cities['City name'][1])

print(type(cities[0:2]))
print(cities[0:2])

## 데이터 조작
# Python의 기본 산술 연산을 Series에 적용 할 수 있습니다.
print(population / 1000.)

# Numpy는 과학적 컴퓨팅을 위한 인기있는 툴킷입니다. pandas Series는 대부분의 Numpy 함수에 대한 인수로 사용할 수 있습니다.
import numpy as np
print(np.log(population))

# 더 복잡한 단일 열 변환을 위해 Series.apply는 강력한 메커니즘을 제공합니다. 각 값에 대해 복잡한 처리를 할 수 있도록 λ 함수 인수를 허용합니다. 아래 예제는 인구가 백만 명이 넘는 지 여부를 나타내는 새로운 Series를 만듭니다.
population.apply(lambda val: val > 1000000)

# DataFrame 수정 또한 간단합니다. 예를 들어 다음 블록은 두 개의 Series를 기존 DataFrame에 추가합니다.
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
display(cities)

## 연습1
# 다음 두 가지 모두가 참인 경우에만 참인 새 부울 열을 추가하여 cities 테이블을 수정하십시오.

 # 도시는 성자 이름을 딴 것입니다.
 # 도시의 면적은 50 평방 마일을 초과합니다.
 # 힌트 : "San"은 스페인어로 "성자"를 의미합니다.

 ## 인덱스
 # Series 및 DataFrame 개체는 행 순서를 제어하는 인덱스 속성도 정의합니다. 기본적으로 생성시 pandas는 원본 데이터의 순서를 반영하는 인덱스를 만듭니다. 생성 된 인덱스 값은 안정적입니다. 다시 말하면 순서가 바뀌지 않습니다.
 print(city_names.index)

 print(list(cities.index))

 # 수동으로 행의 순서를 변경하려면 DataFrame.reindex를 호출하십시오. 예를 들어, 다음은 도시 이름별로 정렬하는 것과 동일한 효과가 있습니다.
 print(cities.reindex([2,0,1]))

 # 다시 색인하는 것은 DataFrame을 섞는 좋은 방법입니다. 아래의 예제에서 우리는 배열과 같은 인덱스를 가져 와서 Numpy의 random.permutation 함수에 전달합니다.이 함수는 값을 제자리에 셔플합니다. 이 shuffled 배열로 reindex를 호출하면 데이터 프레임이 동일한 방식으로 셔플됩니다. 셀을 여러 번 실행 해보십시오!
 display(cities.reindex(np.random.permutation(cities.index)))

 ## 연습2
 # reindex 는 원래 DataFrame의 색인 값에 없는 색인 값을 사용할 수 있습니다. 시도해보고 그런 값을 사용하면 어떻게되는지 확인해 보세요! 왜 이것이 허용될까요?

 ## 그룹화
 # DataFrame을 몇가지 규칙에 따라 그룹화할 수가 있습니다. SQL 과 같이 다양한 group by 명령들을 지원합니다. group by는 지정한 열의 값으로 데이터들을 묶는 기능을 합니다. 붓꽃 데이터로 한 번 확인해봅시다.

import pandas as pd
iris = pd.read_csv("http://datasets.lablup.ai/public/tutorials/iris.csv")
display(iris.head())

print(iris.groupby('species')['sepal_length'].mean())

## 데이터가 빠진 경우
# 데이터를 다루다보면 일부 데이터가 누락된 경우를 흔히 접하게 됩니다. 이러한 경우를 pandas로 처리해 봅시다.

import numpy as np
missing_series = pd.Series([1, 2, 3, np.nan])   # 일부 데이터가 누락된 Series를 만듭니다.
print(missing_series)

print(missing_series.fillna(5))    # 데이터가 누락된 부분을 5로 채웁니다.

print(missing_series.dropna())    # 데이터가 누락된 경우 무시합니다.

