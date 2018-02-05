
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
until = "2017-12-24"

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

# 전체 작업
for i in range(0,len(r['data'])) :
    print(i)
    print("output len :", len(output))
    # post
    a = dictfilt(r['data'][i],('type','id','link','created_time','message','shares'))
    a = pd.DataFrame([a])
    a = a.loc[:,('type','id','link','created_time','message','shares')]
    a.columns.values[1] = 'post_id'

    # reactions
    b = r['data'][i]['reactions']['data']
    b = pd.DataFrame.from_dict(b)
    next = ' ' if 'reactions' not in r['data'][i]['reactions'].keys() else r['data'][i]['reactions']['paging']['next']

    if next == ' ' :
       continue
    else :
       while True :
        try :
            n = json.loads(request_until_suceed(next))
            b1 = pd.DataFrame.from_dict(n['data'])
            b = b.append(b1)
            if 'next' in n['paging'] :
                next = n['paging']['next']
            else : 
                del(next)
                break
        except :
            continue

    del(next)
    
    b.columns = ['user_id','reactions_type']

    ## post + reactions
    post_reactions1 = pd.concat([a,b],axis=0)
    post_reactions1 = post_reactions1.loc[:,('type','post_id','link','created_time','message','shares','user_id','reactions_type')]

    print(len(post_reactions1))
    output = pd.concat([output,post_reactions1])

    # comments
    c = ' ' if 'comments' not in r['data'][i].keys() else r['data'][i]['comments']['data']

    if  c == ' ' :
        continue
    else :
        c = pd.DataFrame.from_dict(c)
        c = c.loc[:,('id','created_time','message')]

        c1 = {'type': ['comments'], 'post_id' : [None],'link': [None] , 'shares' : [None]}

        c1 = pd.DataFrame(data = c1)

        c2 = pd.concat([c1,c],axis=1)
        c2 = c2.loc[:,('type','post_id','link','created_time','message','shares','id')]
        c2.columns.values[6] = 'user_id'

        # comments-reactions
        for k in range(0,len(c2)) : 
            for j in range(0,len(r['data'][i]['comments']['data'])) :
                d = ' ' if 'reactions' not in r['data'][i]['comments']['data'][j].keys() else r['data'][i]['comments']['data'][j]['reactions']['data']
                
                if d == ' ' :
                    d = {'user_id': [None],'reactions_type': [None]}
                    continue
                else :
                    d = pd.DataFrame.from_dict(d)
                    next = r['data'][i]['comments']['data'][j]['reactions']['paging'].get('next')

                    if next is None :
                        continue
                    else :
                        while True :
                            try :
                                n = json.loads(request_until_suceed(next))
                                d1 = pd.DataFrame.from_dict(n['data'])
                                d = d.append(d1)
                                if 'next' in n['paging'] :
                                    next = n['paging']['next']
                                else : 
                                    del(next)
                                    break
                            except :
                                continue
        
                    d.columns = ['user_id','reactions_type']

                c3 = pd.concat([c2.iloc[[k]],d],axis=0)
                c3 = c3.loc[:,('type','post_id','link','created_time','message','shares','user_id','reactions_type')]

                print(len(c3))
                output = pd.concat([output,c3])


    # sharedposts

output['user_id']


####----------------------------
id = '255834461424286' # tensorflowKR
id = '496657243716553' # himart

# group or page feed(post)
url2 = "/255834461424286/feed?fields=sharedposts{sharedposts{sharedposts{sharedposts{sharedposts,message,created_time},message,created_time},message,created_time},message,created_time},message,created_time&since=2017-12-25&until=2017-12-25&access_token=1263444613742411|4e23a6676a157697655619d65c200fb4"

# sharedposts
url2 = "/255834461424286_576695412671521/sharedposts?fields=sharedposts,promotable_id&access_token=1263444613742411|4e23a6676a157697655619d65c200fb4"

####
url2 = "/255834461424286/feed?fields=created_time,link,message,message_tags,type,shares,reactions{type},comments{created_time,message,reactions{type},comments{created_time,message,reactions{type}}},sharedposts{reactions{type},comments{id,created_time,reactions{type}},permalink_url,created_time,message,shares}&since=2017-12-23&until=2017-12-24"

token = '&access_token=1263444613742411|4e23a6676a157697655619d65c200fb4'

url = base + url2 + token
results = json.loads(request_until_suceed(url))

data = []
data.extend(results['data'])

i = 0
while True :
    try :
        r = json.loads(request_until_suceed(results['paging']['next']))
        if r['data'] == [] :
            break
        else :
            data.extend(r['data'])
            i += 1
    except :
        print ("done")
        break


# 255834461424286_576695412671521/sharedposts?fields=from{name,fan_count}
# /page-id/feed?fields=sharedposts{from{name,fan_count}}
# {post_id}/sharedposts?fields=likes.summary(1),id,created_time,from.summary(1)&limit=1500
# 255834461424286/feed?fields=sharedposts{from},shares,id,object_id,reactions&since=2017-12-21&until=2017-12-23

# 컨텐츠 생산된 위치에 맞춰, sharedposts가 나타남. 현재 그룹이나 페이지가 아닌 개인,그룹,페이지에서 생성된 post를 shared할 경우에는 sharedposts가 안 잡힘.

### 1. page 또는 group의 feed(post) 수집
# - created_time
# - link
# - type
# - message
# - message_tags
# - shares
# - reactions
# +- type
# - comments
#  +- id
#  +- created_time
#  +- comments
#   +- id
#   +- created_time
#   +- reactions
#    +- type
#  +- reactions
#   +- type

##--------