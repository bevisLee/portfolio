

# -*- coding: cp949 -*-

import json
import csv
import urllib.request
import pandas as pd

app_id = "1263444613742411"
app_secret = "4e23a6676a157697655619d65c200fb4"
access_token = app_id + "|" + app_secret
page_id = "255834461424286"
since = "2017-12-23"
until = "2017-12-27"

# facebook graph api - https://developers.facebook.com/tools/explorer/
base = "https://graph.facebook.com"
node = "/" + page_id + "/feed"
parameters1 = "?fields=created_time,link,message,message_tags,type,shares,reactions{type},comments{created_time,message,reactions{type},comments{created_time,message,reactions{type}}},sharedposts{reactions{type},comments{id,created_time,reactions{type}},permalink_url,created_time,message,shares}"
time = "&since=%s&until=%s" % (since, until)
access = "&access_token=%s" % access_token

url = base + node + parameters1 + time + access

def request_until_suceed(url):
    req = urllib.request.Request(url)
    success = False
    while success is False:
        try:
            response = urllib.request.urlopen(req)
            if response.getcode() == 200:
                success = True
        except Exception as e:
            print(e)  # wnat to know what error it is
            time.sleep(5)
            print("Error for url %s : %s" % (url, datetime.datetime.now()))

    return response.read().decode(response.headers.get_content_charset())

# retrieve data
r = json.loads(request_until_suceed(url))

dictfilt = lambda x, y: dict([ (i,x[i]) for i in x if i in set(y) ])

output = pd.DataFrame()

# data crawling
for i in range(0,len(r['data'])) :
    print(i)
    print("output len :", len(output))

    # post
    a = dictfilt(r['data'][i],('type','id','link','created_time','message','shares'))
    a = pd.DataFrame([a])
    a = a.loc[:,('type','id','link','created_time','message','shares')]
    a.columns.values[1] = 'post_id'

    # reactions
    b = ' ' if 'reactions' not in r['data'][i].keys() else r['data'][i]['reactions']['data']

    if b != ' ' :
        b = pd.DataFrame.from_dict(b)

        next = ' ' if 'reactions' not in r['data'][i]['reactions'].keys() else r['data'][i]['reactions']['paging']['next']
        if next != ' ' :
           while True :
            try :
                n = json.loads(request_until_suceed(next))
                b1 = pd.DataFrame.from_dict(n['data'])
                b = b.append(b1)
                next = ' ' if 'next' not in n['paging'].keys() else n['paging']['next']
                if next == ' ' :
                    del(next)
                    break
            except :
                print("reactions while finish")
                break

        del(next)
    
        b.columns = ['user_id','reactions_type']

        ## post + reactions
        post_reactions1 = pd.concat([a,b],axis=0)
        post_reactions1 = post_reactions1.loc[:,('type','post_id','link','created_time','message','shares','user_id','reactions_type')]

        print("post_reactions1 len :", len(post_reactions1))
        output = pd.concat([output,post_reactions1])
    else :
        print("a len :", len(a))
        output = pd.concat([output,a])

    # comments
    c = ' ' if 'comments' not in r['data'][i].keys() else r['data'][i]['comments']['data']

    if  c != ' ' :
        c = pd.DataFrame.from_dict(c)
        c = c.loc[:,('id','created_time','message')]

        c1 = {'type': ['comments'], 'post_id' : [None],'link': [None] , 'shares' : [None]}

        c1 = pd.DataFrame(data = c1)

        c2 = pd.concat([c1,c],axis=1)
        c2 = c2.loc[:,('type','post_id','link','created_time','message','shares','id')]
        c2.columns.values[6] = 'user_id'

        # comments-reactions
        for j in range(0,len(r['data'][i]['comments']['data'])) :
            d = ' ' if 'reactions' not in r['data'][i]['comments']['data'][j].keys() else r['data'][i]['comments']['data'][j]['reactions']['data']
                
            if d == ' ' :
                d = {'user_id': [None],'reactions_type': [None]}
                continue
            else :
                d = pd.DataFrame.from_dict(d)
                next = ' ' if 'reactions' not in r['data'][i]['comments']['data'][j]['reactions'].keys() else r['data'][i]['comments']['data'][j]['reactions']['paging']['next']

                if next != ' '  :
                    while True :
                        try :
                            n = json.loads(request_until_suceed(next))
                            d1 = pd.DataFrame.from_dict(n['data'])
                            d = d.append(d1)
                            next = ' ' if 'next' not in n['paging'].keys() else n['paging']['next']
                            if next == ' ' :
                                del(next)
                                break
                        except :
                            print("reactions while finish")
                            break
        
                d.columns = ['user_id','reactions_type']

            c3 = pd.concat([c2.iloc[[j]],d],axis=0)
            c3 = c3.loc[:,('type','post_id','link','created_time','message','shares','user_id','reactions_type')]
            c3['type'] = 'comments'

            print("c3 len:", len(c3))
            output = pd.concat([output,c3])

    # sharedposts
    e = ' ' if 'sharedposts' not in r['data'][i].keys() else r['data'][i]['sharedposts']['data']

    if e != ' ' :
        e = pd.DataFrame.from_dict(e)
        e['type'] = 'sharedposts'
        e = e.loc[:,('type','id','permalink_url', 'created_time','message','shares')]
        e.columns.values[1] = 'post_id'
        e.columns.values[2] = 'link'

        # reactions
        for l in range(0,len(r['data'][i]['sharedposts']['data'])) :
            f = ' ' if 'reactions' not in r['data'][i]['sharedposts']['data'][l].keys() else r['data'][i]['sharedposts']['data'][l]['reactions']['data']

            if f != ' ' :
                f = pd.DataFrame.from_dict(f)
                f.columns = ['user_id','reactions_type']

                f2 = pd.concat([e.iloc[[l]],f],axis=0)
                f2 = f2.loc[:,('type','post_id','link','created_time','message','shares','user_id','reactions_type')]
                f2['type'] = 'sharedposts'

                # next??議댁옱?섎굹, data媛 ?쒓났?섏? ?딆븘 ?앸왂
                print("f2 len :", len(f2))
                output = pd.concat([output,f2])

        # comments
        for m in range(0,len(r['data'][i]['sharedposts']['data'])) :
            g = ' ' if 'comments' not in r['data'][i]['sharedposts']['data'][m].keys() else r['data'][i]['sharedposts']['data'][l]['comments']['data']

            if g != ' ' :
                g = pd.DataFrame.from_dict(g)
                g = g.loc[:,('id','created_time','message')]
                g1 = {'type': ['sharedposts_comments'], 'post_id' : [None],'link': [None] , 'shares' : [None]}

                g1 = pd.DataFrame(data = g1)

                g2 = pd.concat([g1,g],axis=1)
                g2 = g2.loc[:,('type','post_id','link','created_time','message','shares','id')]
                g2.columns.values[6] = 'user_id'
                g2['type'] = 'sharedposts_comments'

                print("g2 len :", len(g2))
                output = pd.concat([output,g2])

## output csv save
output = output.loc[:,('type','post_id','link','created_time','message','shares','user_id','reactions_type')]

output.to_csv(('C:/Users/bevis/Downloads/work_daum_news/'+'facebook_page_post_%s_%s.csv' % (since, until)), index=False) 

###---- to do

## 1
# output['shares'] data type = dict 이라, value만 남기는 작업이 필요
# 아니면, 엑셀에서 수동으로 제거가 필요

## 2
# network analytics 수료 후, user_id 간 network 분석 진행 필요
# https://campus.datacamp.com/courses/network-analysis-in-python-part-1/introduction-to-networks?ex=1

## 3
# 수집된 user_id와 facebook ad나 pixel에 user_id와 맵핑할 수 있는 방법이 있는지 확인 필요
# 수집된 user_id 기준 profile 조회하여, email 이나 mobile number 수집할 수 있는지 확인 필요