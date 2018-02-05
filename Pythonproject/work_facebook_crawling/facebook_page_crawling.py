
"""
Created on 01/11/2017

@author: bevis
"""

"""
Facebook Fan page 수집기 

수집 대상 : 페이스북 api

1. 특정 Fan Page는 검색 결과에 있는 id 중  Fan이 가장 많은 페이지를 기준으로 수집
2. 특정 Fan Page 검색 후 name, id, link, fan 저장 

- access_token은 token 인증 기간이 있어, 실행이 안되는 경우 toekn 갱신이 필요
- 페이지 이름에 국가명, 등 불필요한 정보는 제외하여 사용
""" 

####------------------------------------------------------
# 페이스북 팬페이지 리스트 입력
####------------------------------------------------------
import pandas as pd 
import pandas as DataFrame

with open("E:/work_facebookpage/fanpage_171106.txt", "rb") as f:
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

# 참고 페이지 : https://developers.facebook.com/tools/explorer/
# 기본적으로 token은 만료기간이 1일이라, 실행이 안되는 경우 참고 페이지에서 만료일 갱신 필요
# 한번에 많이 처리하면 token이 block 당하여, token을 새로 받아서 진행해야함

access_token = 'EAACEdEose0cBAI7M31ihPHGb8EUCsvDokqJZBqk6nbv0OZB9smnXcxSuZClwA53AcvJp0u0MS5ZAo8LVx60ozl5pD2mFgO0ZCV6OzP0dh4KzZB5BvBt3eCPaLcDLaxSFI7ZBZCP2W6OwmsoJa4N7XmLZA7yrUhGCyMP3aZApy2ZBlwa58AeLaP7OkoToSi5tQ0DINxbZCx3DkG8HZAwZDZD' # token 교체 예정

####------------------------------------------------------
# 페이스북 팬페이지 URL 수집
####------------------------------------------------------

graph = facebook.GraphAPI(access_token)

# output = pd.DataFrame()

for i in range(4155,len(fbpage)): # for i in range(0,len(fbpage)):
    print(("---------------------------------------------------------------------"))

    page_keyword = []     # 입력한 페이지 name 
    page_name = []        # 페이지 name 
    page_url = []         # 페이지 url 
    page_fan = []         # 페이지 fan
    page_id = []          # 페이지 id

    keyword = fbpage.iloc[i]["list"]
    keyword = keyword.replace(u'\xa0', u' ')
    page_keyword.append(keyword)
    print((str(i)+"번째(page_keyword) 수집 완료 : " + str(keyword)))

    data = graph.request('/search?q='+keyword+'&type=page')

    if (i % 10) == 0 : ## 10  배수인 경우 10초 후에 수집(token block 방지용)
        time.sleep(10) # token block 경우 사용   
        if data['data'] == [] :
            name = "NA"
            page_name.append(name)
            id = "NA"
            page_id.append(id)
            link = "NA"
            page_url.append(link)
            fan = "NA"
            page_fan.append(fan)
            print((str(i)+"번째(All) : 입력 키워드 검색 결과 없음"))

            row = pd.DataFrame()
            row['keyword'] = page_keyword
            row['name'] = page_name
            row['page_url'] = page_url
            row['page_fan'] = page_fan
            row['page_id'] = page_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_page_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Page save complted : All NA"))
            print(("---------------------------------------------------------------------"))
        else : 
            fan_num = []
            for j in range(0, len(data['data'])) :
                a = data['data'][j]['id']
                c = graph.get_object(id = a, fields='fan_count')
                d = c['fan_count']
                fan_num.append(d)

            index = fan_num.index(max(fan_num))
            list = data['data'][index]

            name = list['name']
            page_name.append(name)
            id = list['id']
            page_id.append(id)
            print((str(i)+"번째(page_id) 수집 완료 : " + str(id)))

            link = graph.get_object(id = id, fields='link')
            link = link['link']
            page_url.append(link)
            print((str(i)+"번째(page_url) 수집 완료 : " + str(link)))

            # facebook graph ver에 따라 'likes' 또는 'fan_count'를 맞춰 변경해서 사용해야함
            fan = graph.get_object(id = id, fields='fan_count')
            fan = fan['fan_count']
#            fan = graph.get_object(id = id, fields='likes')
#            fan = fan['likes']
            page_fan.append(fan)
            print((str(i)+"번째(page_fan) 수집 완료 : " + str(fan)))
            
            row = pd.DataFrame()
            row['keyword'] = page_keyword
            row['name'] = page_name
            row['page_url'] = page_url
            row['page_fan'] = page_fan
            row['page_id'] = page_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_page_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Page save complted : All OK"))
            print(("---------------------------------------------------------------------"))

    else :  ## 10  배수가 아닌 경우에는 빠르게 수집
        if data['data'] == [] :
            name = "NA"
            page_name.append(name)
            id = "NA"
            page_id.append(id)
            link = "NA"
            page_url.append(link)
            fan = "NA"
            page_fan.append(fan)
            print((str(i)+"번째(All) : 입력 키워드 검색 결과 없음"))

            row = pd.DataFrame()
            row['keyword'] = page_keyword
            row['name'] = page_name
            row['page_url'] = page_url
            row['page_fan'] = page_fan
            row['page_id'] = page_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_page_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Page save complted : All NA"))
            print(("---------------------------------------------------------------------"))
        else : 
            fan_num = []
            for j in range(0, len(data['data'])) :
                a = data['data'][j]['id']
                c = graph.get_object(id = a, fields='fan_count')
                d = c['fan_count']
                fan_num.append(d)

            index = fan_num.index(max(fan_num))
            list = data['data'][index]

            name = list['name']
            page_name.append(name)
            id = list['id']
            page_id.append(id)
            print((str(i)+"번째(page_id) 수집 완료 : " + str(id)))

            link = graph.get_object(id = id, fields='link')
            link = link['link']
            page_url.append(link)
            print((str(i)+"번째(page_url) 수집 완료 : " + str(link)))

            # facebook graph ver에 따라 'likes' 또는 'fan_count'를 맞춰 변경해서 사용해야함
            fan = graph.get_object(id = id, fields='fan_count')
            fan = fan['fan_count']
#            fan = graph.get_object(id = id, fields='likes')
#            fan = fan['likes']
            page_fan.append(fan)
            print((str(i)+"번째(page_fan) 수집 완료 : " + str(fan)))
            
            row = pd.DataFrame()
            row['keyword'] = page_keyword
            row['name'] = page_name
            row['page_url'] = page_url
            row['page_fan'] = page_fan
            row['page_id'] = page_id
            output = pd.concat([output, row])
            output.to_csv('E:/work_facebookpage/fb_page_output.csv', index=False) # 디렉토리 변경 필요
            print(("output Page save complted : All OK"))
            print(("---------------------------------------------------------------------"))
