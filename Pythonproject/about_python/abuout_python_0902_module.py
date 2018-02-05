from __future__ import print_function

def sum(a,b):
    return a+b

sum(1,2)

### if 

money = 1

if money :
        print("택시를 타고 가라")
else :
        print("걸어 가라")

# 자료형	참	거짓
# 숫자	0이 아닌 숫자	0
# 문자열	"abc"	""
# 리스트	[1,2,3]	[]
# 터플	(1,2,3)	()
# 딕셔너리	{"a":"b"}	{}

# 비교연산자	설명
# x < y	x가 y보다 작다
# x > y	x가 y보다 크다
# x == y	x와 y가 같다
# x != y	x와 y가 같지 않다
# x >= y	x가 y보다 크거나 같다
# x <= y	x가 y보다 작거나 같다

money = 2000

if money >= 3000 :
    print("택시를 타고 가라")
else :
    print("걸어가라")

# 연산자	설명
# x or y	x와 y 둘중에 하나만 참이면 참이다
# x and y	x와 y 모두 참이어야 참이다
# not x	x가 거짓이면 참이다

money = 2000
card = 1

if money >=3000 or card :
    print("택시를 타고 가라")
else :
    print("걸어가라")

# in	not in
# x in 리스트	x not in 리스트 [ ]
# x in 튜플	x not in 튜플 ( )
# x in 문자열	x not in 문자열 { }

1 in [1, 2, 3] # True
1 not in [1, 2, 3] # False

pocket = ["paper", "money", "cellphone"]

if "money" in pocket :
    pass
else :
    print("카드를 꺼내라")

pocket = ["paper", "cellphone"]
card = 1

if "money" in pocket :
    print("택시를 타고 가라")
else :
    if card :
        print("택시를 타고 가라")
    else : 
        print("걸어가라")

if "money" in pocket :
    print("현금을 사용하여 택시를 타고 가라")
elif card :
    print("카드를 사용하여 택시를 타고 가라")
else :
    print("걸어가라")

### while

treeHit = 0
while treeHit < 10 :
    treeHit = treeHit + 1
    print("나무를 %d번 찍었습니다." % treeHit)
    if treeHit == 10 :
        print("나무가 넘어갔습니다.")

number = 0

prompt = print("1. Add\n2. Del\n3. List\n4. Quit")

while number != 4 :
    print(prompt) # prompt 에러 / ipython에 내장된 함수인지 확인 필요
    number = int(input())


coffee = 10
money = 300

while money :
    print("돈을 받았으니 커피를 줍니다.")
    coffee = coffee -1
    sales = (10-coffee)*300
    print("현재 남은 커피의 양은 %d 개입니다." % coffee)
    print("현재 커피의 매출액은 %d 원입니다." % sales)
    if not coffee :
        print("커피가 다 떨어졌습니다. 판매를 중지합니다.")
        print("최종 커피의 매출액은 3000 원입니다.")
        break

coffee = 10
while True:
    money = int(input("돈을 넣어 주세요: "))
    if money == 300:
        print("커피를 줍니다.")
        coffee = coffee -1
    elif money > 300:
        print("거스름돈 %d를 주고 커피를 줍니다." % (money -300))
        coffee = coffee -1
    else:
        print("돈을 다시 돌려주고 커피를 주지 않습니다.")
        print("남은 커피의 양은 %d개 입니다." % coffee)
    if not coffee:
        print("커피가 다 떨어졌습니다. 판매를 중지 합니다.")
        break

## coffee 수량에 따라 수정이 필요

coffee = 10

while coffee != 0 :
    money = int(input("커피를 주문하실려면 돈을 넣어주세요."))
    if money < 300 :
        print("300원 보다 적게 돈을 넣었습니다. 300원 이상 넣어주세요. 넣으신 돈은 반납하였습니다.")
    elif money >= 300 :
         number = int(input("주문할 커피의 수량을 입력하세요"))
         refund = money - number*300
         print("주문한 커피는 %d 잔이고, 잔돈은 %d 입니다." %(number,refund))
         coffee = coffee - number
         print("남은 커피의 양은 %d개 입니다." % coffee)
    if not coffee:
         print("커피가 다 떨어졌습니다. 판매를 중지 합니다.")
         break
