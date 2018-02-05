
number = 1

while number :
    if number < 101 :
        if number % 2 == 0 :
            print("%s" % number)
            number = number + 1
        else :
            number = number + 1
    else :
        break

## Q1_answer start
nums = (1,3) # range(1,101) 1 ~ 101


## Q1_answer end

select = "test"

while select != "q" :
   select = input("아무 말이나 입력하세요. 종료는 q 입력 : ")
   if select == "q" :
      print("종료를 입력하셨습니다. 종료합니다.")
      break
   else :
      print("%s" % select)

## Q2_answer start

done = False
while not done :
    keyval = input("아무말이나(종료를 원하면 q) : ")
    print(keyval)
    if keyval == "q" :
        done = True

## Q2_answer end


raw_src = '''Trying to let you know
Sign을 보내 signal 보내
I must let you know
Sign을 보내 signal 보내
Sign을 보내 signal 보내
Sign을 보내 signal 보내
Sign을 보내 signal 보내
I must let you know
Sign을 보내 signal 보내
근데 전혀 안 통해
눈빛을 보내 눈치를 주네
근데 못 알아듣네
답답해서 미치겠다 정말
왜 그런지 모르겠다 정말
다시 한 번 힘을 내서
Sign을 보내 signal 보내
눈짓도 손짓도 어떤 표정도
소용이 없네 하나도 안 통해
눈치도 코치도 전혀 없나 봐
더 이상 어떻게 내 맘을 표현해
언제부턴가 난 네가 좋아
지기 시작했어 바보야
왜 이렇게도 내 맘을 몰라
언제까지 이렇게 둔하게
나를 친구로만 대할래
내가 원하는 건 그게 아닌데
Signal 보내 signal 보내
찌릿 찌릿 찌릿 찌릿
난 너를 원해 난 너를 원해
왜 반응이 없니
만날 때 마다 마음을 담아
찌릿 찌릿 찌릿 찌릿
기다리잖아 다 보이잖아
왜 알지 못하니
Trying to let you know
Sign을 보내 signal 보내
I must let you know
Sign을 보내 signal 보내
널 보며 웃으면 알아채야지
오늘만 몇 번째 널 보며 웃는데
자꾸 말을 걸면 좀 느껴야지
계속 네 곁에 머물러있는데
언제부턴가 난 네가 좋아
지기 시작했어 바보야
왜 이렇게도 내 맘을 몰라
언제까지 이렇게 둔하게
나를 친구로만 대할래
내가 원하는 건 그게 아닌데
Signal 보내 signal 보내
찌릿 찌릿 찌릿 찌릿
난 너를 원해 난 너를 원해
왜 반응이 없니
만날 때 마다 마음을 담아
찌릿 찌릿 찌릿 찌릿
기다리잖아 다 보이잖아
왜 알지 못하니
찌릿 찌릿 찌릿 찌릿
왜 반응이 없니
찌릿 찌릿 찌릿 찌릿
왜 알지 못하니
Sign을 보내 signal 보내
근데 전혀 안 통해
눈빛을 보내 눈치를 주네
근데 못 알아듣네
답답해서 미치겠다 정말
왜 그런지 모르겠다 정말
다시 한 번 힘을 내서
Sign을 보내 signal 보내'''

## Q3_answer start

doc = raw_src

def count_keyword(doc, keyword) :
    return keyword in doc

## Q3_answer end

cnt = 0
keyword = 'signal'
fn = 'signal.txt'
doc = load_data_from_file(fn)
cnt = count_keyword(doc, keyword)
print(cnt)

## for, 함수, 클래스, 파일 다루기, 웹 다운로드

# Q4_과일 바구니에서 - 사과, 사과, 배, 바나나, 바나나, 배, 배, 딸기, 수박 순서로 과일을 꺼내서 출력

baguni = ["사과", "사과", "배", "바나나", "바나나", "배", "배", "딸기", "수박"]

for n in range(0,len(baguni)) :
    print("%s" % baguni[n])

# Q5_'시그널'이란 단어를 입력받으면 '보내'라는 단어를 출력하는 함수 만들기

text = "test"
def myprint(text) :
    while text != "q" :
        text = input("signal 을 입력하세요. 종료는 q 입력 : ")
        if text == "signal" :
          print("보내~ 보내~")
          break

myprint(text)

## Q5_answer start

def myprint(text, num) :
    print(text*num)

text = "시그널"
myprint("시그널",3)

## Q5_answer end

# Q6_MyNum 이라는 클래스를 만들어서 one, two, three 인스턴스를 생성. 이것들을 다 더해서 출력하기

class dog() :
    def __init__(self, name) :
        self.name = name
    def wal(self) :
        print(self.name) # member variable

a = dog("happy")

class Mynum() :
    def __init__(self, value) :
        self.value = value
    def info(self) :
        return print(self.value) # member variable

one = Mynum(1)
two = Mynum(2)
three = Mynum(3)

one.info()
two.info()
three.info()

one.value

print(one.value + two.value + three.value)

# Q7_시그널 가사 파일 읽어서 '보내'라는 말만 출력하기

f = open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/about_python/signal.txt", encoding = "utf-8")

def check_bonae(word) :
    if word == "보내" :
        print(word)

def print_bonae(line) :
    words = line.split()
    for word in words :
        check_bonae(word)

for line in f :
    print(line.strip())


# Q8_웹에서 시그널 가사 불러와서 저장하기

f_w = open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/about_python/mysave.txt", "w") # 기본은 python home로 저장
f_w.close()

import re
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent' : 'Mozilla/5.0 (Windows; U; Windows NT 5.1;)'}
r = requests.get('http://cmusic.tistory.com/498', timeout=300, headers=headers)

soup = BeautifulSoup(r.text, 'lxml')

signal_text = ''.join([line.text for line in soup.select('div.tt_article_useless_p_margin p')][3:])

len(re.findall('signal',signal_text))