

## 함수 help?
help(len)

len?

## 단축키
L = [1,2,3]
L. ## .tab -> 객체에 적용할 함수 목록 출력

# ctr + P 위 화살표 : 명령어 히스토리 역순 검색
# ctr + N 아래 화살표 : 명령어 히스토리 최근순 검색
# ctr + R : readline 명령어 형식의 히스토리검색
# Ctr+shift+V : 클립보드에 텍스트 붙여넣기
# ctr + C : 실행중인 코드 중단
# ctr + A : 커서를 줄의 처음으로 이동
# ctr + E : 커서를 줄의 끝으로 이동
# ctr + K : 커서가 놓인 곳부터 줄의 끝까지 텍스트 삭제
# ctr + U : 현재 입력된 모든 텍스트 지우기
# ctr + F : 커서를 앞으로 한글자씩 이동하기
# ctr + B : 커서를 뒤로 한글자씩 이동하기
# ctr + L : 화면 지우기

writefile fibonacci.py
def fibonacc(n):
    if n<1:
        return 1
    else :
        return fibonacc(n-1)+fibonacc(n-2)

from fibonacci import * # fibonacci.py 모든 모듈 실행

fibonacc(10)

import timeit
timeit.timeit fibonacc(10) ## 실행 시간 확인

## History input & output
import math

math.sin(2)

math.cos(2)

prting(__) # 이전 -2 명령 출력
prting(___) # 이전 -3 명령 출력

!ls  / !dir # ipython 상태에서 cmd 명령어 실행

def func1(a,b):
    return a/b

def func2(x):
    a = x
    b = x-1
    return func1(a,b)

func2(1)

def A():
    print("world")

def B():
    print("hello")


from guppy import hpy ## 메모리 관리 패키지 --> 에러


