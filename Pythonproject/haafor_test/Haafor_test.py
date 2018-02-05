
# Q1) 
string = 'abcdeeeooauiea'

def itemgetter(*items):
    if len(items) == 1:
        item = items[0]
        def g(obj):
            return obj[item]
    else:
        def g(obj):
            return tuple(obj[item] for item in items)
    return g

def second_freq(str) :
    frequency = {}

    for word in str :
        count = frequency.get(word,0)
        frequency[word] = count + 1

    tup = tuple(sorted(frequency.items(), key=itemgetter(1), reverse=True))
    print(tup[1][0])
    del(frequency,tup)

second_freq(string)

# Q2)
num = []

while True:
    add = int(input("into any number [exit : q ] : "))
    num.append(add)
    if num == "q" :
        print("입력을 종료합니다..")
        break

num.sort(key=lambda x:((x % 2), -x))
print(num)

# Q3)
stock_price = {'0': 10, '1': 11, '2': 12, '3':13, '4':12, '5':10, '6':9, '7':13, '8':14, '9':15, '10':18,
     '11':19, '12':20, '13':22, '14':23, '15':20, '16':17, '17':16, '18':19, '19':24, '20':23}

index = []
for i in range(0,len(stock_price)-1) :
    if list(stock_price.values())[i] - list(stock_price.values())[i+1] < 0 :
        index.append(i)

index

mdd = []
for i in range(0,len(index)) :
    if index[i] - index[i+1] != -1 :
        mdd.append(index[i]+1)

mdd

MDD = []
for i in mdd : 
   for j in range(i+1, len(stock_price)) :
       if list(stock_price.values())[i] <= list(stock_price.values())[j] :
            a = ((j)-(i),j)
            MDD.append((a))
            MDD = sorted(MDD, key=lambda tup : tup[1], reverse=True)
            break
       else :
            continue

MDD
print(MDD[0])

 
####-----------------------------------------------------
## M1
import math

M = []
for i in range(1, 10) :
    a = float(8)
    b = float(i)
    c = float(6)
    d = float(10-i)
   
    x = (a*a) + (b*b)
    y = (c*c) + (d*d)

    z = math.sqrt(x) + math.sqrt(y)
    z1 = (z,i)
    M.append(z1)

min(M) = (17.21110255092798, 6)

# 최단 이동거리 : 17.21110255092798

## M2
M = []
for i in range(1, 25) :
    a = float(10)
    b = float(i)
    c = float(10)
    d = float(25-i)
   
    x = (a*a) + (b*b)
    y = (c*c) + (d*d)

    z = math.sqrt(x) + math.sqrt(y)/2
    z1 = (z,i)
    M.append(z1)

min(M) = (22.360679774997898, 5)

# 최단시간 이동거리 : 22.360679774997898

## M3 
a = 2/6 * 1/5 = 2/30 = 1/15
b = 2/6 * 4/5 = 8/30 = 4/15
c = 4/6 * 3/5 * 2/4 * 1/3 = 30/360 = 1/15

## E3
# 이자보상비율 = 영업이익 / 이자비용
    # 영업이익 = 매출액 - 판매원가 - 판관비
= 1000 - 500 - 300 / 50 = 200 / 50 = 4

# 순이익 = 매출액 - 매출원가 - 판관비 및 일반관리비 - 영업외이익,영업외비용 - 특별이익,특별손실,법인세비용 - (사업손실)
= 1000 - 500 - 300 - 50 - 50 - 10 = 90