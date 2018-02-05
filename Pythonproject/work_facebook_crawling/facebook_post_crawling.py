
"""
Created on 01/11/2017

@author: bevis
"""

"""
Facebook Fan post 수집기 

수집 대상 : 페이스북 api

1. 저장된 Fan Page 정보를 기준으로 마지막 발행일, 발행주기 계산하여 저장

- access_token은 token 인증 기간이 있어, 실행이 안되는 경우 toekn 갱신이 필요
- 페이지 이름에 국가명, 등 불필요한 정보는 제외하여 사용
""" 

####------------------------------------------------------
# 페이스북 팬페이지 리스트 입력
####------------------------------------------------------
import pandas as pd 
import pandas as DataFrame

## csv 파일에 utf-8 에러 발생시 txt에 name,pageid 저장하여 사용
with open("E:/work_facebookpage/fb_page_output_chlee.txt", "rb") as f:
    contents = f.read().decode("UTF-8")

contents = contents.split('\r\n')

fbpage = pd.DataFrame(contents)
fbpage.columns =["page_id"]

## csv 파일에 utf-8 저장한 경우 사용
fbpage = pd.read_table('E:/work_facebookpage/fb_page_output2.csv',sep=',')
fbpage = fbpage.dropna() # NaN 행 제거

####------------------------------------------------------
# 페이스북 인증키 입력
####------------------------------------------------------
import facebook
import requests
import re
from datetime import datetime
import urllib
import time

# 참고 페이지 : https://developers.facebook.com/tools/explorer/
# 기본적으로 token은 만료기간이 1일이라, 실행이 안되는 경우 참고 페이지에서 만료일 갱신 필요
# 한번에 많이 처리하면 token이 block 당하여, token을 새로 받아서 진행해야함

access_token = 'EAAR9GJ9lP0sBAGoktRaG2f1m61ZAKH9i9ZCFZCLYJ9McTTGB0CxblcpFzPglLPUS0A7hQFPyGc6LemI0WN3urhjJUye7CGN3lHbW0bwZC8LngyA37IAm81qStU880YwzjOVBoE6AZBxyvsLJBF1nLMFTwTnPOOVYZD' # token 교체 예정


####------------------------------------------------------
# 페이스북 팬페이지 URL 수집
####------------------------------------------------------

graph = facebook.GraphAPI(access_token)

output = pd.DataFrame()

for i in range(0,len(fbpage)):
    print(("---------------------------------------------------------------------"))

    page_name = []        # 페이지 name
    page_id = []          # 페이지 id
    post_last = []        # 마지막 발행일
    post_period = []      # 발행 주기

    id = str(int(fbpage.iloc[i]["page_id"]))
    page_id.append(id)
    name = ""
    page_name.append(name)
    print((str(i)+"번째(name) 수집 : " + str(name)))
    print((str(i)+"번째(post_id) 수집 : " + str(id)))

    if (i % 100) == 0 :
        time.sleep(10) # token 튕길 경우 사용   
    ####------------------------------------------------------
    # 페이스북 팬페이지 post 수집
    ####------------------------------------------------------
        profile = graph.get_object(id)
        posts = graph.get_connections(profile['id'], 'posts')

        ## 발행된 post가 없는 경우, 공백으로 처리
        if len(posts['data']) == 0 : 
            b = "NA"
            post_last.append(b)
            period = "NA"
            post_period.append(period)
            print((str(i)+"번째(post_last,period) : " + " 검색 결과 없음"))

            row = pd.DataFrame()
            row['page_name'] = page_name
            row['page_id'] = page_id
            row['post_last'] = post_last
            row['post_period'] = post_period
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_post_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Post save complted : 2 NA"))
            print(("---------------------------------------------------------------------"))
        elif len(posts['data']) == 1 :
            post_time = []
            a = posts['data'][0]
            b = a['created_time']
            b = re.sub('T.*','',b)
            post_last.append(b)
            c = datetime.strptime(b , '%Y-%m-%d')
            post_time.append(c)
            print((str(i)+"번째(post_last) 수집 완료 : " + str(b)))

            d = datetime.today()
            period = d - c
            period = period.days
            post_period.append(period)
            print((str(i)+"번째(post_period) 수집 완료 : " + str(period)))

            row = pd.DataFrame()
            row['page_name'] = page_name
            row['page_id'] = page_id
            row['post_last'] = post_last
            row['post_period'] = post_period
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_post_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Post save complted : post 1"))
            print(("---------------------------------------------------------------------"))
        else :
            post_time = []
            for j in range(0, len(posts['data'])) :
                a = posts['data'][j]
                b = a['created_time']
                b = re.sub('T.*','',b)
                if j == 0 :
                    post_last.append(b)
                    c = datetime.strptime(b , '%Y-%m-%d')
                    post_time.append(c)
                    print((str(i)+"번째(post_last) 수집 완료 : " + str(b)))
                else :
                    c = datetime.strptime(b , '%Y-%m-%d')
                    post_time.append(c)

            post_period_1 = []
            for k in range(0, len(post_time)-1) :
                a = post_time[k] - post_time[k+1] 
                a = a.days
                post_period_1.append(a)

            period = sum(post_period_1)/len(post_period_1)
            post_period.append(period)
            print((str(i)+"번째(post_period) 수집 완료 : " + str(period)))

            row = pd.DataFrame()
            row['page_name'] = page_name
            row['page_id'] = page_id
            row['post_last'] = post_last
            row['post_period'] = post_period
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_post_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Post save complted : All OK"))
            print(("---------------------------------------------------------------------"))
    else :
        profile = graph.get_object(id)
        posts = graph.get_connections(profile['id'], 'posts')

        ## 발행된 post가 없는 경우, 공백으로 처리
        if len(posts['data']) == 0 : 
            b = "NA"
            post_last.append(b)
            period = "NA"
            post_period.append(period)
            print((str(i)+"번째(post_last,period) : " + " 검색 결과 없음"))

            row = pd.DataFrame()
            row['page_name'] = page_name
            row['page_id'] = page_id
            row['post_last'] = post_last
            row['post_period'] = post_period
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_post_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Post save complted : 2 NA"))
            print(("---------------------------------------------------------------------"))
        elif len(posts['data']) == 1 :
            post_time = []
            a = posts['data'][0]
            b = a['created_time']
            b = re.sub('T.*','',b)
            post_last.append(b)
            c = datetime.strptime(b , '%Y-%m-%d')
            post_time.append(c)
            print((str(i)+"번째(post_last) 수집 완료 : " + str(b)))

            d = datetime.today()
            period = d - c
            period = period.days
            post_period.append(period)
            print((str(i)+"번째(post_period) 수집 완료 : " + str(period)))

            row = pd.DataFrame()
            row['page_name'] = page_name
            row['page_id'] = page_id
            row['post_last'] = post_last
            row['post_period'] = post_period
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_post_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Post save complted : post 1"))
            print(("---------------------------------------------------------------------"))
        else :
            post_time = []
            for j in range(0, len(posts['data'])) :
                a = posts['data'][j]
                b = a['created_time']
                b = re.sub('T.*','',b)
                if j == 0 :
                    post_last.append(b)
                    c = datetime.strptime(b , '%Y-%m-%d')
                    post_time.append(c)
                    print((str(i)+"번째(post_last) 수집 완료 : " + str(b)))
                else :
                    c = datetime.strptime(b , '%Y-%m-%d')
                    post_time.append(c)

            post_period_1 = []
            for k in range(0, len(post_time)-1) :
                a = post_time[k] - post_time[k+1] 
                a = a.days
                post_period_1.append(a)

            period = sum(post_period_1)/len(post_period_1)
            post_period.append(period)
            print((str(i)+"번째(post_period) 수집 완료 : " + str(period)))

            row = pd.DataFrame()
            row['page_name'] = page_name
            row['page_id'] = page_id
            row['post_last'] = post_last
            row['post_period'] = post_period
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_post_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Post save complted : All OK"))
            print(("---------------------------------------------------------------------"))

