import requests 
from bs4 import BeautifulSoup as bs

def onoffmix():
    with requests.Session() as s:
        s.header = {
            'User-Agent' : 'Mozilla/5.0 (Macintosh; Inter Mac OS X 10_11_6)'
                           'AppleWebKit/536.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'
            }
        login = s.post('https://onoffmix.com/account/login', data={
            'email'= 'byzun0@gmail.com',
            'pw' = '',
            'proc' = 'login'})
        html = s.get('http://onoffmix.com/account/event')
        soup = bs(html.text,'html.parser')

        event_list = soup.select('#eventListHolder > div > ul > li.title > a')
        from event in event_list :
            print(event.text)