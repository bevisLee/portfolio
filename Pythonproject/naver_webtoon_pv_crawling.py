
from selenium import webdriver

main_url = 'http://comic.naver.com/genre/challenge.nhn?&page='

## 셀레늄 사용 시작: 크롬창 켜짐, 실행 끝날 때까지 닫으면 안됨 
driver = webdriver.Chrome(executable_path=r'C:/Users/bevis/Downloads/chromedriver_win32/chromedriver.exe')

## 1. 도전만화 리스트 웹툰 url 수집
sub_url = []
for i in range(1,101) :
    print(i,'번째 수집중....', round((i/101)*100,1),"% 완료")
    page = i
    url = main_url+str(page)
    driver.get(url)

    elems = driver.find_elements_by_xpath("//a[@href]")

    sub_url1 = []
    for elem in elems:
        sub_url1.append(elem.get_attribute("href"))

    # sub_url 에서 id
    for nn in range(len(sub_url1)) :
        if 'titleId' in sub_url1[nn] :
            sub_url.append(sub_url1[nn])

    sub_url = list(set(sub_url))
    print('수집된 sub_url :',len(sub_url))

## 2. 수집된 리스트 웹툰별 회차 url 수집
sub_url_list=[]
for i in range(2038,len(sub_url)) :
    print(i,'번째 수집중....', round((i/len(sub_url))*100,1),"% 완료")
    driver.get(sub_url[i])

    elems = driver.find_elements_by_xpath("//a[@href]")

    sub_url1 = []
    for elem in elems:
        sub_url1.append(elem.get_attribute("href"))

    for nn in range(len(sub_url1)) :
            if str(sub_url[i][len(sub_url[i])-6:])+'&no' in sub_url1[nn] :
                sub_url_list.append(sub_url1[nn])
                break

    sub_url_list = list(set(sub_url_list))
    print('수집된 sub_url_list :',len(sub_url_list))

## 3. 웹툰 세부페이지에서 웹툰명, 연재명, 조회수 저장
import re

webtoon_list = []
for i in range(0,len(sub_url_list)) :
    print(i,'번째 추출중....', round((i/len(sub_url_list))*100,1),"% 완료")
    try :
        if 35 < int(re.findall('\d+', sub_url_list[i][len(sub_url_list[i])-3:])[0]) :
            webtoon_list.append(sub_url_list[i])
    except:
        continue

import pandas as pd
import time

output = pd.DataFrame()
output['webtoon'] = None
output['sub_title'] = None
output['score'] = None
output['date'] = None
output['pv'] = None 

for i in range(99,len(webtoon_list)) :
    print(i,'번째 수집중....', round((i/len(webtoon_list))*100,1),"% 완료")

    num = int(re.findall('\d+', webtoon_list[i][len(webtoon_list[i])-3:])[0])

    webtoon = []
    sub_title = []
    score = []
    date = []
    pv = []
    url = webtoon_list[i][:62]

    try :
        for j in range(1,num+1) :
            web_url = url+str(j)
            driver.get(web_url)
            time.sleep(1)

            nt = driver.find_element_by_class_name('detail')
            name1 = nt.text
            name1 = name1.split('\n')
            name1 = name1[0]
            webtoon.append(name1)

            nt1 = driver.find_element_by_class_name('view')
            sb1 = nt1.text
            sb1 = sb1.split('\n')
            sb1 = sb1[0]
            sub_title.append(sb1)

            st = driver.find_elements_by_class_name('total')
            rt1 = st[0].text

            rt = driver.find_elements_by_class_name('date')
            rt2 = rt[0].text
            rt3 = rt[1].text
            score.append(rt1)
            date.append(rt2)
            pv.append(rt3)
    except:
        continue

    output1 = pd.DataFrame()
    output1['webtoon'] = webtoon
    output1['sub_title'] = sub_title
    output1['score'] = score
    output1['date'] = date
    output1['pv'] = pv

    output = output.append(output1)

driver.quit()

output.to_csv('C:/Users/bevis/Downloads/naverwebtoon_180114.csv', index=False)

# 웹툰명, 날짜 기준으로 내림차순 정렬 / 웹툰, 연재명, 날짜 기준 그룹핑
output = output.sort_values(['webtoon', 'date'],ascending=False).groupby(['webtoon', 'sub_title','date'])