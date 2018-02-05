

####------------------------------------------------------
# 페이스북 팬페이지 리스트 입력
####------------------------------------------------------
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as DataFrame

with open("C:/Users/bevis/Downloads/work_facebookpage/fanpage_171031.txt", "rb") as f:
    contents = f.read().decode("UTF-8")

contents = contents.split('\r\n')

fbpage = pd.DataFrame(contents)
fbpage.columns =["list"]

####------------------------------------------------------
# 페이스북 팬페이지 URL 수집
####------------------------------------------------------

import requests
import urllib
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from selenium import webdriver
import re

import time
import random

title = []        #  지역
page_url = []     # 페이지 url 
page_id = []      # 페이지 id
like = []         # 좋아요 

driver = webdriver.Chrome(executable_path=r'C:/Users/bevis/Downloads/chromedriver_win32/chromedriver.exe')

for i in range(0, len(fbpage)):
    print (fbpage.iloc[i])
    keyword = fbpage.iloc[i]["list"]
  
    encoded_args = urllib.parse.quote(keyword.encode('utf-8'))
    u = ('https://m.facebook.com/search/pages/?q='+encoded_args+'&source=filter&isTrending=0')
    driver.get(u)

    soup = BeautifulSoup(urllib.request.urlopen(u).read())

    time.sleep(random.randint(3, 9))

    lists = soup.select('td.bc')
    if lists == [] : 
       lists = soup.select('td.q')

    li = lists[0]
    page = li.select_one('a')

    link = page.attrs['href']
    link = re.sub('/\?.*','',link)
    link = re.sub('/','',link)
    page_id.append(link)
    link = ("https://facebook.com/"+link)
    page_url.append(link)
        
    name = page.text
    title.append(name)

    info = li.select_one('div')
    num = info.text
    num = re.sub('[^0-9]','',num)
    like.append(num)
    print('- URL 총 '+str(i)+' 건 수집중 ')

### 112개에 bot 체크 들어옴
### bot 체크로 중단 : for i range(0, 에서 0을 중단된 숫자부터 수정해서 ip 변경 후 재실행

output = pd.DataFrame()

output['title'] = title
output['page_url'] = page_url
output['page_id'] = page_id
output['like'] = like

output.to_csv('C:/Users/bevis/Downloads/work_facebookpage/fb_page_output.csv', index=False) # 디렉토리 변경 필요