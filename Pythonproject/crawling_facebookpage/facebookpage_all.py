"""
Created on 01/11/2017

@author: bevis
"""

"""
Facebook Fan page 수집기 

수집 대상 : 다음 뉴스 

1. 특정 Fan Page 검색 후 name, id, link, fan 저장
2. 저장된 Fan Page 정보를 기준으로 마지막 발행일, 발행주기 계산하여 저장

- access_token은 token 인증 기간이 있어, 실행이 안되는 경우 toekn 갱신이 필요
- 페이지 이름에 국가명, 등 불필요한 정보는 제외하여 사용
""" 

####------------------------------------------------------
# 페이스북 팬페이지 리스트 입력
####------------------------------------------------------
import pandas as pd
import pandas as DataFrame

with open("C:/Users/bevis/Downloads/work_facebookpage/fanpage_171031.txt", "rb") as f:
    contents = f.read().decode("UTF-8")

contents = contents.split('\r\n')

fbpage = pd.DataFrame(contents)
fbpage.columns =["list"]

####------------------------------------------------------
# 페이스북 인증키 입력
####------------------------------------------------------
import facebook
import requests
import re
from datetime import datetime
import urllib
from dateutil.parser import parse

# import time
# import random

# 참고 페이지 : https://developers.facebook.com/tools/explorer/
# 기본적으로 token은 만료기간이 1일이라, 실행이 안되는 경우 참고 페이지에서 만료일 갱신 필요
# access_token = 'EAACEdEose0cBAANBpwvyZCvbaRL2mjPguxa88PvL2SoLprgCIpxsTUjTckjDSpRAPdzMQnaPLtoDZCcExw94CPQavjzOaxU3CRWCtTKoIcmZC1h3cCctvPu1qW194yeaQGslNa9ZATMVaIFL9XLodVbAmlyKYZBaMEJmUWtx0WBNgzYRqHwwtCqDxgcg2ycZBk6tirLn0aZBlO3nEXJZCEhecqMvjgWmQY0ZD'

access_token = 'EAAR9GJ9lP0sBAL8wJT4YJSkbMCvG2Yvsx8QdMJkr3fIbk4XjwopyqcWkZAmC2awqcTSu6AtFeCZBfyMT71Qjzpr03a7JnYbNc0UkMXHR02VEJLeh0Ahi4zeLM23tYRMaPlhl5p4mI4VFhLjziBcI4fyWBqF6YZD' # token 교체 예정

####------------------------------------------------------
# 페이스북 팬페이지 URL 수집
####------------------------------------------------------
page_keyword = []     # 입력한 페이지 name 
page_name = []        # 페이지 name 
page_url = []         # 페이지 url 
page_fan = []         # 페이지 fan
page_id = []          # 페이지 id
post_last = []        # 마지막 발행일
post_period = []      # 발행 주기

for i in range(0, len(fbpage)):

    print(("---------------------------------------------------------------------"))

    keyword = fbpage.iloc[i]["list"]
    keyword = keyword.replace(u'\xa0', u' ')
    page_keyword.append(keyword)
    print((str(i)+"번째(page_keyword) 수집 완료 : " + str(keyword)))
    encoded_args = urllib.parse.quote(keyword.encode('utf-8'))

    graph = facebook.GraphAPI(access_token)

 #   time.sleep(random.randint(2, 6)) # token 튕길 경우 사용

    data = graph.request('/search?q='+encoded_args+'&type=page')

    # page 검색 결과가 없는 경우, 공백으로 처리
    if data['data'] == [] :
        name = "NA"
        page_name.append(name)
        id = "NA"
        page_name.append(name)
        link = "NA"
        page_url.append(link)
        fan = "NA"
        page_fan.append(fan)
        print((str(i)+"번째(All) : 입력 키워드 검색 결과 없음"))
    else : 
        list = data['data'][0]

        name = list['name']
        page_name.append(name)
        id = list['id']
        page_id.append(id)
        print((str(i)+"번째(page_id) 수집 완료 : " + str(id)))
        name = ""

        link = graph.get_object(id = id, fields='link')
        link = link['link']
        page_url.append(link)
        print((str(i)+"번째(page_url) 수집 완료 : " + str(link)))
        link = ""

        fan = graph.get_object(id = id, fields='fan_count')
        fan = fan['fan_count']
    #        fan = graph.get_object(id = id, fields='likes')
    #        fan = fan['likes']
        page_fan.append(fan)
        print((str(i)+"번째(page_fan) 수집 완료 : " + str(fan)))
        fan = ""
####------------------------------------------------------
# 페이스북 팬페이지 post 수집
####------------------------------------------------------
        profile = graph.get_object(id)
        posts = graph.get_connections(profile['id'], 'posts')

        ## 발행된 post가 없는 경우, 공백으로 처리
        if posts['data'] == [] : 
            b = "NA"
            post_last.append(b)
            period = "NA"
            post_period.append(period)
            print((str(i)+"번째(post_last,period) : " + " 검색 결과 없음"))
        else :
            post_time = []
            if len(posts['data']) > 1 :
                for  j in range(0, len(posts['data'])) :
                    a = posts['data'][j]
                    b = a['created_time']
                    b = re.sub('T.*','',b)
                    if j == 0 :
                        post_last.append(b)
                        c = datetime.strptime(b , '%Y-%m-%d')
                        post_time.append(c)
                        print((str(i)+"번째(post_last) 수집 완료 : " + str(c)))
                    else :
                        c = datetime.strptime(b , '%Y-%m-%d')
                        post_time.append(c)

                post_period = []
                for k in range(0, len(post_time)-1) :
                    a = post_time[k] - post_time[k+1] 
                    a = a.days
                    post_period.append(a)

                period = sum(post_period)/len(post_period)
                post_period.append(period)
                print((str(i)+"번째(post_period) 수집 완료 : " + str(period)))
                print(("---------------------------------------------------------------------"))
            else :
                a = posts['data'][j]
                b = a['created_time']
                b = re.sub('T.*','',b)
                post_last.append(b)
                c = datetime.strptime(b , '%Y-%m-%d')
                post_time.append(c)
                print((str(i)+"번째(post_last) 수집 완료 : " + str(c)))

                d = datetime.today()
                period = d - c
                period = period.days
                post_period.append(period)
                print((str(i)+"번째(post_period) 수집 완료 : " + str(period)))
                print(("---------------------------------------------------------------------"))
                 

####------------------------------------------------------
# 페이스북 팬페이지 수집된 정보 저장
####------------------------------------------------------

output = pd.DataFrame()

output['name'] = page_name
output['page_url'] = page_url
output['page_fan'] = page_fan
output['page_id'] = page_id
output['post_last'] = post_last
output['post_period'] = post_period

output

output.to_csv('C:/Users/bevis/Downloads/work_facebookpage/fb_page_output.csv', index=False) # 디렉토리 변경 필요
