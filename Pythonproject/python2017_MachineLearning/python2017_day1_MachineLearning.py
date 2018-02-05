import tensorflow as tf

### Numpy 기본

import numpy as np # numpy를 np라는 약어로 불러옵니다. 앞으로 계속 사용하겠습니다.

# Lablup.AI의 라이브러리에서 display 명령을 불러옵니다.
# display는 테이블 형식의 결과를 웹에서 보기 좋게 포매팅해주는 명령입니다.
# 중요한 부분이 아니니 무시해도 됩니다.

# from sorna.display import display
from IPython.display import display

## 배열

# numpy 배열은 모두 같은 유형의 값 그리드이며 음수가 아닌 정수의 튜플로 인덱싱됩니다. 차원의 수는 배열의 랭크입니다. 배열의 모양은 각 차원을 따라 배열의 크기를 제공하는 정수의 튜플입니다.
# 중첩 된 파이썬 리스트로부터 numpy 배열을 초기화 할 수 있고 대괄호를 사용하여 요소에 접근 할 수 있습니다.

a = np.array([1,2,3]) # 랭크 1 배열을 만듭니다
print(type(a),a.shape,a[0],a[1],a[2])

a[0] = 5 # 배열의 요소를 변경합니다

display(a)
# sorna.display error로 실행 결과 다름
# ---------------------------------------------------------------------------
# ModuleNotFoundError                       
# Traceback (most recent call last)
# ~\AppData\Local\Temp\~vs3E02.py in <module>()
# ----> 1 from sorna.display import display

# ModuleNotFoundError: No module named 'sorna.display'

b = np.array([[1,2,3],[4,5,6]]) # 랭크 2 배열을 만듭니다

display(b)

print(b.shape) # b의 모양을 출력합니다. 일반적으로 배열의 모양은 배열의 차원입니다.

print(b[0,0],b[0,1],b[1,0])

# Numpy는 또한 배열을 생성하는 많은 함수를 제공합니다.

a = np.zeros((2,2)) # 0으로 가득찬 배열을 만듭니다.
display(a)

b = np.ones((1,2)) # 1로 가득찬 배열을 만듭니다.
display(b)

c = np.full((2,2),7) # 상수 배열을 만듭니다.
display(c)

d = np.eye(2) # 2x2 단위행렬을 만듭니다.
display(d)

e = np.random.random((2,2)) # 0~1 사이의 값 중 무작위값으로 채워진 배열을 만듭니다. 
display(e)

f = np.random.randn(2,2) # 표준정규분포를 따르는 무작위값 배열을 만듭니다.
display(f)

g = np.random.normal(size=(2,2)) #정규분포를 따르는 무작위값 배열을 만듭니다.
display(g)

# 배열 색인 생성하기

# Numpy는 배열에 색인을 생성하는 몇 가지 방법을 제공합니다.
# 슬라이싱 : 파이썬 리스트와 마찬가지로, numpy 배열은 슬라이스 될 수 있습니다. 배열은 다차원일 수 있으므로 배열의 각 차원에 대해 슬라이스를 지정해야합니다.

# (3,4) 형태의 랭크 2 배열을 만듭니다.
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

# 슬라이싱을 이용하여 처음 두 행과 두 열로 구성된 부분배열을 만듭니다.
# b 는 (2, 2) 형태의 배열이 됩니다.
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
display(b)

# 배열의 조각은 동일한 데이터에 대한 뷰이므로 이를 수정하면 원래 배열이 수정됩니다.

print(a[0,1])
b[0,0] = 77  # b[0, 0] 는 a[0, 1] 과 동일한 데이터를 갖고 있습니다.
print(a[0,1])

# 정수 인덱싱과 슬라이스 인덱싱을 함께 사용할 수도 있습니다. 그러나 이렇게하면 원래 배열보다 낮은 랭크의 배열이 생성됩니다. 이것은 MATLAB이 배열 슬라이스를 처리하는 방식과 매우 다릅니다.

# (3, 4) 모양의 랭크 2 배열을 만듭니다.
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
display(a)

# 배열의 중간 행에 있는 데이터에 액세스하는 두 가지 방법이 있습니다. 정수 인덱스와 슬라이스를 섞으면 더 낮은 랭크의 배열이 생성되는 반면, 슬라이스만 사용하면 원래 배열과 동일한 랭크의 배열이 생성됩니다.
row_r1 = a[1,:] # a의 두번째 행의 랭크 1뷰
row_r2 = a[1:2,:] # a의 두번째 행의 랭크 2뷰
row_r3 = a[[1],:] # a의 두번째 행의 랭크 2뷰
print(row_r1, row_r1.shape)  # 정수 인덱스와 섞은 경우
print(row_r2, row_r2.shape)  # 슬라이스만 사용한 경우
print(row_r3, row_r3.shape)  # 슬라이스만 사용한 경우

# 정수 배열 인덱싱 : 슬라이싱을 사용하여 numpy 배열로 인덱싱 할 때 결과 배열 뷰는 항상 원래 배열의 하위 배열이 됩니다. 반대로 정수 배열 인덱싱을 사용하면 다른 배열의 데이터를 사용하여 임의의 배열을 구성 할 수 있습니다. 다음은 그 예입니다.
a = np.array([[1,2],[3,4],[5,6]])

# 정수 배열 인덱싱 예.
# 결과 배열은 (3,) 형태가 됩니다. 
display(a[[0, 1, 2], [0, 1, 0]])

# 위의 정수 배열 인덱싱은 아래와 같습니다.
display(np.array([a[0, 0], a[1, 1], a[2, 0]]))

# 위의 정수 배열 인덱싱 예와 같은 경우입니다.
display(np.array([a[0,1],a[0,1]]))

# 정수 배열 인덱싱을 사용하는 유용한 트릭은 행렬의 각 행에서 하나의 요소를 선택하거나 변경하는 것입니다.
# 요소를 선택할 새 배열을 만듭니다.
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
display(a)

# 인덱스 배열을 만듭니다.
b = np.array([0, 2, 0, 1])

# b의 인덱스를 이용하여 a의 각 행으로부터 하나의 요소들을 선택합니다.
display(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# b의 인덱스를 이용하여 각 행의 하나의 요소를 변경합니다.
a[np.arange(4), b] += 10
display(a)

# 부울 배열 인덱싱 : 부울 배열 인덱싱을 사용하면 배열의 임의 요소를 선택할 수 있습니다. 이 유형의 인덱싱은 일부 조건을 만족하는 배열 요소를 선택하는 데 자주 사용됩니다. 다음은 그 예입니다.
a = np.array([[1,2],[3,4],[5,6]])
bool_idx = (a > 2)
# 2보다 큰 a의 원소를 찾습니다.
# 이것은 a와 동일한 모양의 부울 값 배열을 반환합니다.
# 여기서 bool_idx의 각 슬롯은 a의 해당 요소가 2보다 큰지 여부를 나타냅니다.
display(bool_idx)

# 부울 배열 인덱싱을 사용하여 bool_idx의 True 값에 해당하는 요소로 구성된 
# 랭크 1 배열을 생성합니다
display(a[bool_idx])

# 위의 모든 것을 간결하게 할 수 있습니다.
display(a[a > 2])

# 간결함을 위해 numpy 배열 인덱싱에 대한 많은 세부 사항을 생략했습니다. 더 많은 것을 알고 싶다면 문서를 읽어야합니다.

## 데이터 유형
# 각 numpy 배열은 동일한 유형의 요소 그리드입니다. Numpy는 배열을 생성하는 데 사용할 수있는 많은 수의 데이터 유형을 제공합니다. Numpy는 배열을 생성 할 때 데이터 유형을 추측하려 하지만 배열을 생성하는 함수는 대개 데이터 유형을 명시적으로 지정하는 선택적 인수를 포함합니다. 다음은 그 예입니다.

x = np.array([1, 2])  # numpy 가 데이터 유형을 선택하게 합시다.
y = np.array([1.0, 2.0])  # numpy 가 데이터 유형을 선택하게 합시다.
z = np.array([1, 2], dtype=np.int64)  # 특정 데이터 유형을 강제로 할당합시다.

print(x.dtype, y.dtype, z.dtype)

# numpy 데이터 유형 - https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html

## 배열 수학
# 기본 수학 함수는 배열에서 요소 단위로 작동하며 연산자 오버로드와 numpy 모듈의 함수로 사용할 수 있습니다.

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# 요소별 합. 둘 다 배열을 만듭니다.
display(x + y)
display(np.add(x, y))

# 요소별 차. 둘 다 배열을 만듭니다.
print(x - y)
print(np.subtract(x, y))

# 요소별 곱. 둘 다 배열을 만듭니다.
print(x * y)
print(np.multiply(x, y))

# 요소별 나누기. 둘 다 배열을 만듭니다.
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# 요소별 제곱근. 둘 다 배열을 만듭니다.
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

# MATLAB과 달리 *는 행렬 곱셈이 아닌 요소 별 곱셈입니다. 대신에 벡터의 내적을 계산하고, 벡터에 행렬을 곱하고, 행렬을 곱하기 위해 dot 함수를 사용합니다. dot 는 numpy 모듈의 함수 및 배열 객체의 인스턴스 메서드로 사용할 수 있습니다.
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# 벡터의 내적.  둘 다 219
print(v.dot(w))
print(np.dot(v, w))

# 행렬 / 벡터 곱. 둘 다 랭크 1 배열 [29 67] 을 만듭니다.
print(x.dot(v))
print(np.dot(x, v))

# 행렬 / 행렬 곱. 둘 다 랭크 2 배열을 만듭니다.
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

# Numpy는 배열에서 계산을 수행하는 데 유용한 많은 기능을 제공합니다. 가장 유용한 것 중 하나가 sum입니다.
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # 모든 요소의 합. "10"을 출력합니다.
print(np.sum(x, axis=0))  # 각 행의 합. "[4 6]" 을 출력합니다.
print(np.sum(x, axis=1))  # 각 열의 합. "[3 7]" 을 출력합니다.

# 여기서 축(axis)은 각 배열의 차원에 해당되는 인덱스입니다. 위의 예를 설명하면

# x.shape 은 (2, 2) 입니다.
# np.sum(x, axis=0) 은
# x.shape[axis]: x.shape[0] 에 대하여 연산을 하라는 의미입니다.

# X.shape == (5, 3, 2) 인 경우를 생각해봅시다. 이 경우 np.sum(X, axis=1) 의 결과값은

# 1. X.shape[axis] => X.shape[1] 에 대해서 연산을 하기 때문에
# 2. np.sum(X, axis=1).shape 은 (5, 3, 2) -> (5, 2) 가 됩니다.
# numpy가 제공하는 수학 함수의 전체 목록은 문서에서 찾을 수 있습니다.

# 배열을 사용하여 수학 함수를 계산하는 것 외에도 배열의 데이터를 변형하거나 조작해야 하는 경우가 자주 있습니다. 이 유형의 연산 중 가장 간단한 예는 행렬을 교차(transpose)하는 것입니다. 행렬을 교차하려면 단순히 배열 객체의 T 속성을 사용하십시오.
print(x)
print(x.T)

v = np.array([[1,2,3]])
print(v)
print(v.T)

## 브로드캐스팅
# 브로드캐스팅은 numpy가 산술 연산을 수행 할 때 다른 모양의 배열로 작업 할 수있게 해주는 강력한 메커니즘입니다. 종종 더 작은 배열과 더 큰 배열이 있을 때 더 작은 배열을 여러 번 사용하여 더 큰 배열에서 어떤 연산을 수행하기를 원할 때가 있습니다.
# 예를 들어, 행렬의 각 행에 상수 벡터를 추가하려 한다고 가정합시다. 다음과 같이 할 수 있습니다.

# 행렬 x의 각 행에 벡터 v를 더하고 그 결과를 행렬 y에 저장할 것입니다.
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # x와 같은 모양의 빈 행렬 생성

# 루프를 이용하여 행렬의 각 행에 벡터 v를 더한다.
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

# 이것도 돌아갑니다. 그러나 행렬 x가 매우 클 경우 Python에서 명시적 루프를 계산하는 속도가 느려질 수 있습니다. 행렬 x의 각 행에 벡터 v를 더하는 것은 v의 여러 복사본을 수직으로 쌓은 다음 행렬 x와 v의 요소별 합계를 수행하여 행렬 v를 형성하는 것과 같습니다. 이 접근법은 다음과 같이 구현할 수 있습니다.
vv = np.tile(v, (4, 1))  # v의 4개의 카피를 쌓음.
print(vv)                # 출력은   "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"

y = x + vv  # x 와 vv를 요소별로 더함
print(y)

# Numpy 브로드캐스팅은 실제로 v의 여러 복사본을 만들지 않고도 이 계산을 수행 할 수 있게 해줍니다. 브로드캐스팅을 사용한 경우를 봅시다.
import numpy as np

# 행렬 x의 각 행에 벡터 v를 더하고 그 결과를 행렬 y에 저장할 것입니다.
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # 브로드캐스팅을 이용해 x의 각 행에 v를 더하기
print(y)

# y = x + v 라인은 브로드캐스팅으로 인해 x가 shape (4, 3)이고 v가 shape (3)인데도 작동합니다. 이 행은 v가 실제로 shape (4, 3)인 것처럼 작동합니다. 각 행은 v의 사본이었고, 합계는 요소별로 수행되었습니다.

# 두 개의 배열을 브로드캐스팅하는 것은 다음 규칙을 따릅니다.
# 1. 배열의 랭크가 같지 않으면 두 모양이 같은 길이가 될 때까지 배열의 낮은 랭크쪽에 1을 붙입니다.
# 2. 두 배열은 차원에서 크기가 같거나 배열 중 하나의 차원에 크기가 1 인 경우 차원에서 호환 가능하다고 합니다.
# 3. 배열은 모든 차원에서 호환되면 함께 브로드캐스트 될 수 있습니다.
# 4. 브로드캐스트 후 각 배열은 두 개의 입력 배열의 요소 모양 최대 개수와 동일한 모양을 가진 것처럼 동작합니다.
# 5. 한 배열의 크기가 1이고 다른 배열의 크기가 1보다 큰 차원에서 첫 번째 배열은 마치 해당 차원을 따라 복사 된 것처럼 작동합니다

# 실제로 동작하는 방식을 생각해 봅시다.

# 1. A와 B의 모양을 생각합니다.
# 2. 두 배열이 len(A.shape) == len(B.shape)인지 확인을 합니다.
# 3. 같지 않은 경우에는 두 배열의 모양 길이가 같아질때까지 적은 쪽의 shape 앞에 1 을 추가해 줍니다.
 # 예: (5,3)–>(1,5,3)
# 4. shape이 1인 곳은 복사가 됩니다.
 # 예: shape의 변화는 아래와 같게 될겁니다.
 # (5, 3)+(3,)
 # (5, 3)+(1, 3)
 # (5, 3) + (5, 3)
 # (5, 3)
 # 브로드캐스팅 - https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

 # 벡터의 외적을 계산합니다.
v = np.array([1,2,3])  # v 는 (3,) 모양
w = np.array([4,5])    # w 는 (2,) 모양
# 외적을 계산하기 위해서, v를 (3, 1) 모양의 열벡터로 모양을 변경합니다; 
# 그런 다음 w에 대해서 브로드캐스트를 하여 v와 w의 외적인 (3, 2) 모양의 결과를 얻습니다:

print(np.reshape(v, (3, 1)) * w)

# 벡터를 행렬의 각 행에 더합니다.
x = np.array([[1,2,3], [4,5,6]])
# x 는 (2, 3) 모양이며, v 는 (3, ) 모양입니다. 그래서 (2, 3)으로 브로드캐스트됩니다:

print(x + v)

# 행렬의 각 열에 벡터를 추가합니다. 
# x는 (2, 3) 모양이고 w는 (2,) 모양 입니다.
# x를 전치한다면 (3, 2) 모양이 되며, w에 대해 브로드캐스트하여 (3, 2) 모양의 결과를 얻을 수 있습니다. 
# 이 결과를 전치하면 (2, 3) 모양의 최종 결과가 나오는데, 이는 행렬 x의 각 열에 벡터 w를 더한 것입니다.

print((x.T + w).T)

# 다른 방법은 w의 모양을 (2, 1)의 행벡터로 바꾸는 것입니다.
# 그 다음 x에 대해 바로 브로드캐스트하여 같은 결과를 만들 수 있습니다.
print(x + np.reshape(w, (2, 1)))

# 행렬에 상수를 곱합니다:
# x 는 (2, 3)의 모양입니다. Numpy 는 스칼라를 () 모양의 배열로 다룹니다;
# 이것들은 아래의 방법으로 (2, 3) 모양으로 브로드캐스트할 수 있습니다.
print(x * 2)

# 브로드캐스팅을 이용해 일반적으로 코드를 보다 간결하고 빠르게 작성할 수 있으므로 가능하면 브로드캐스팅을 사용하려고 노력해야합니다.
# numpy 레퍼런스 - https://docs.scipy.org/doc/numpy/reference/
