

"""
Created on 01/11/2017

@author: bevis
"""

"""
Facebook User 수집기 

수집 대상 : 페이스북 api

1. 특정 User 검색 후, 이름, id, link 수집

- access_token은 token 인증 기간이 있어, 실행이 안되는 경우 toekn 갱신이 필요
""" 

####------------------------------------------------------
# 페이스북 팬페이지 리스트 입력
####------------------------------------------------------
import pandas as pd 
import pandas as DataFrame

# txt 파일 생성할때, 목록 기입 후 다른이름 저장으로하여, 인코딩을 'UTF-8'을 선택하여 저장
with open("E:/work_facebookpage/reporter.txt", "rb") as f:
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
import time
# from dateutil.parser import parse

# 참고 페이지 : https://developers.facebook.com/tools/explorer/
# 기본적으로 token은 만료기간이 1일이라, 실행이 안되는 경우 참고 페이지에서 만료일 갱신 필요
# 한번에 많이 처리하면 token이 block 당하여, token을 새로 받아서 진행해야함

access_token = 'EAACEdEose0cBAAiXgSKPMvjxUM33tveMe1oT30koxikiolDG1HTMp8hdAyWiWz0tyQ8y52tXFvjKw4eUoVouaCeeO4KZAwvaAE3OnkPsuXREWoBWB9liw4qIdJf2lz35Nxkey84QUipAm7vtUk0ZCdXpx3ODHAYkjHNVSiiR3Oco3jCEdsuBCWznnDiakHQYNnb0l1QQZDZD' # token 교체 예정

####------------------------------------------------------
# 페이스북 팬페이지 URL 수집
####------------------------------------------------------

graph = facebook.GraphAPI(access_token)

output = pd.DataFrame() 
for i in range(0,len(fbpage)):
    print(("---------------------------------------------------------------------"))

    user_keyword = []     # 입력한 기자 이름 & 소속 
    user_name = []        # 검색된 기자 name 
    user_url = []         # 검색된 기자 url 
    user_id = []          # 검색된 기자 id

    keyword = fbpage.iloc[i]["list"]
    keyword = keyword.replace(u'\xa0', u' ')
    print((str(i)+"번째(user_keyword) 수집 완료 : " + str(keyword)))

    data = graph.request('/search?q='+keyword+'&type=user')

    if (i % 10) == 0 : ## 10  배수인 경우 10초 후에 수집(token block 방지용)
        time.sleep(10) # token block 경우 사용   
        if data['data'] == [] :
            user_keyword.append(keyword)
            name = "NA"
            user_name.append(name)
            id = "NA"
            user_id.append(id)
            link = "NA"
            user_url.append(link)
            print((str(i)+"번째(All) : 입력 키워드 검색 결과 없음"))

            row = pd.DataFrame()
            row['keyword'] = user_keyword
            row['name'] = user_name
            row['user_url'] = user_url
            row['user_id'] = user_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_reporter_output.csv', index=False) # 디렉토리 변경 필요
            print(("output User save complted : All NA"))
            print(("---------------------------------------------------------------------"))
        else : 
            for j in range(0, len(data['data'])) :
                user_keyword.append(keyword)

                list = data['data'][j]

                name = list['name']
                user_name.append(name)
                print(str(i)+"번째 (keyword) 검색결과 : "+(str(j)+"번째(user_name) 수집 완료 : " + str(name)))

                id = list['id']
                user_id.append(id)
                print(str(i)+"번째 (keyword) 검색결과 : "+(str(j)+"번째(user_id) 수집 완료 : " + str(id)))

                link = graph.get_object(id = id, fields='link')
                link = link['link']
                user_url.append(link)
                print(str(i)+"번째 (keyword) 검색결과 : "+(str(j)+"번째(user_url) 수집 완료 : " + str(link)))
            
            row = pd.DataFrame()
            row['keyword'] = user_keyword
            row['name'] = user_name
            row['user_url'] = user_url
            row['user_id'] = user_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_reporter_output.csv', index=False) # 디렉토리 변경 필요
            print(("output User save complted : All OK"))
            print(("---------------------------------------------------------------------"))

    else :  ## 10  배수가 아닌 경우에는 빠르게 수집
        if data['data'] == [] :
            user_keyword.append(keyword)
            name = "NA"
            user_name.append(name)
            id = "NA"
            user_id.append(id)
            link = "NA"
            user_url.append(link)
            print((str(i)+"번째(All) : 입력 키워드 검색 결과 없음"))

            row = pd.DataFrame()
            row['keyword'] = user_keyword
            row['name'] = user_name
            row['user_url'] = user_url
            row['user_id'] = user_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_reporter_output.csv', index=False) # 디렉토리 변경 필요
            print(("output User save complted : All NA"))
            print(("---------------------------------------------------------------------"))
        else : 
            for j in range(0, len(data['data'])) :
                user_keyword.append(keyword)

                list = data['data'][j]

                name = list['name']
                user_name.append(name)
                print(str(i)+"번째 (keyword) 검색결과 : "+(str(j)+"번째(user_name) 수집 완료 : " + str(name)))

                id = list['id']
                user_id.append(id)
                print(str(i)+"번째 (keyword) 검색결과 : "+(str(j)+"번째(user_id) 수집 완료 : " + str(id)))

                link = graph.get_object(id = id, fields='link')
                link = link['link']
                user_url.append(link)
                print(str(i)+"번째 (keyword) 검색결과 : "+(str(j)+"번째(user_url) 수집 완료 : " + str(link)))
            
            row = pd.DataFrame()
            row['keyword'] = user_keyword
            row['name'] = user_name
            row['user_url'] = user_url
            row['user_id'] = user_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_reporter_output.csv', index=False) # 디렉토리 변경 필요
            print(("output User save complted : All OK"))
            print(("---------------------------------------------------------------------"))

