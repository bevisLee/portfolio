### 1.3. 파이썬 인터프리터

## 1.3.1. 산술 연산
1 - 2
4 * 5
7/5
3 ** 2

## 1.3.2. 자료형 : data type
type(10) # int(정수)
type(2.719) # float(실수)
type("hello") # str(문자열)

## 1.3.3. 변수 : variable
x = 10 # 초기화
print(x) # x의 값 출력

x = 100 # 변수에 값 대입
print(x)

y = 3.14
x * y

type(x * y)

## 1.3.4. 리스트 : list
a = [1,2,3,4,5] # 리스트 생성
print(a) # 리스트의 내용 출력

len(a) # 리스트의 길이 출력

a[0] # 첫 원소에 접근

a[4] # 다섯번째 원소에 접근 

a[4] = 99 # 다섯번째 원소에 값 대입

print(a)

a[0:2] # 인덱스 0부터 2까지 얻기 (2번째는 포함하지 않는다)

a[1:] # 인덱스 1부터 끝까지

a[:3] # 처음부터 인덱스 3까지 얻기
