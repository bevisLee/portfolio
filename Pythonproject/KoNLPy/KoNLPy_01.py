## 참고 https://www.slideshare.net/kimhyunjoonglovit/pycon2017-koreannlp

from konlpy.tag import Komoran

komoran = Komoran()

print(komoran.nouns(u"[스페셜경제=유민주 기자]오는 31일 한국은행 금융통화위원회 본회의에서 기준 금리를 결정하기 때문에 관련 시장에서 관심이 고조되고 있다. 26일 금융권에 따르면 한국은행은 31일 8월 금융통화위원회 본 회의를 열고 현재 1.25%인 기준 금리의 변동 여부를 결정한다. 한은은 지난달 금통위에서 기준 금리를 1.25%수준에서 13개월째 동결했다. 앞서 이주열 총재가 지난달부터 꾸준히 금리인상 시그널을 보냈지만 시장은 이번달 금통위가 대내외 경제 변수를 좀 더 지켜보기 위해 금리를 동결할 것으로 관측이 우세하다. 우선 금융권은 최근 경기 회복세가 주춤하는 모습을 보여 좀 더 상황을 지켜보자는 판단을 내릴 것이란 예상하고 있다. 경제협력개발기구(OECD)에 따르면 지난 6월 한국의 경기선행지수(CLI)는 100.57이다. 전달(100.60) 대비 0.03포인트 하락했다. 이 지수는 6~9개월 뒤 경기흐름을 예측하는 지표로 3개월째 내림세다. 작년 말부터 이어진 경기 개선 흐름이 약해진 상황 가운데, OECD 경기선행지수도 하락세를 지속하면서 우리 경제의 불확실성이 다시 불거지는 모양새다. 또한 경기회복세를 이끌었던 수출 증가세 그 폭이 급격하게 둔화되고 있다. 한국은행이 발표한 7월 수출물량지수(139.42)는 전년동월대비 0.1% 올라 그 폭이 미미했다. 하지만 북한의 지정학적 리스크와 부동산 대책 등 경기 둔화 요인이 나타나고 있기 때문에 상황이 변했다. 최근 발표된 정부의 8 ·2 부동산대책 발표 이후 거래량이 대폭 줄면서, 주택경기 악화로 건설투자에 타격을 줄 것이란 우려가 커지고 있는 상태다. 올 2분기까지 건설투자의 성장기여도는 5분기 연속 50% 이상을 차지할 정도로 국내 경기에 큰 영향을 주고 있다. 다만 금융권은 한은 기준금리 인상 시점을 일단 내년으로 미뤄질 것으로 보고 있다. 국제금융센터에 따르면 해외 주요 투자은행 9곳 중 7곳은 내년 상반기 우리나라 기준금리가 1.50%로 0.25%포인트 오를 걸로 전망했다. 또한 NH투자증권도 국내외 경기 불확실성 등으로 한은이 연내 기준금리 인상에 나설 가능성이 작다고 봤다. 한편 하나금융투자는 정부의 가계부채 대책의 정책공조를 근거로 연내 인상할 가능성이 있다고 내다봤다."))

print(komoran.pos(u"[스페셜경제=유민주 기자]오는 31일 한국은행 금융통화위원회 본회의에서 기준 금리를 결정하기 때문에 관련 시장에서 관심이 고조되고 있다. 26일 금융권에 따르면 한국은행은 31일 8월 금융통화위원회 본 회의를 열고 현재 1.25%인 기준 금리의 변동 여부를 결정한다. 한은은 지난달 금통위에서 기준 금리를 1.25%수준에서 13개월째 동결했다. 앞서 이주열 총재가 지난달부터 꾸준히 금리인상 시그널을 보냈지만 시장은 이번달 금통위가 대내외 경제 변수를 좀 더 지켜보기 위해 금리를 동결할 것으로 관측이 우세하다. 우선 금융권은 최근 경기 회복세가 주춤하는 모습을 보여 좀 더 상황을 지켜보자는 판단을 내릴 것이란 예상하고 있다. 경제협력개발기구(OECD)에 따르면 지난 6월 한국의 경기선행지수(CLI)는 100.57이다. 전달(100.60) 대비 0.03포인트 하락했다. 이 지수는 6~9개월 뒤 경기흐름을 예측하는 지표로 3개월째 내림세다. 작년 말부터 이어진 경기 개선 흐름이 약해진 상황 가운데, OECD 경기선행지수도 하락세를 지속하면서 우리 경제의 불확실성이 다시 불거지는 모양새다. 또한 경기회복세를 이끌었던 수출 증가세 그 폭이 급격하게 둔화되고 있다. 한국은행이 발표한 7월 수출물량지수(139.42)는 전년동월대비 0.1% 올라 그 폭이 미미했다. 하지만 북한의 지정학적 리스크와 부동산 대책 등 경기 둔화 요인이 나타나고 있기 때문에 상황이 변했다. 최근 발표된 정부의 8 ·2 부동산대책 발표 이후 거래량이 대폭 줄면서, 주택경기 악화로 건설투자에 타격을 줄 것이란 우려가 커지고 있는 상태다. 올 2분기까지 건설투자의 성장기여도는 5분기 연속 50% 이상을 차지할 정도로 국내 경기에 큰 영향을 주고 있다. 다만 금융권은 한은 기준금리 인상 시점을 일단 내년으로 미뤄질 것으로 보고 있다. 국제금융센터에 따르면 해외 주요 투자은행 9곳 중 7곳은 내년 상반기 우리나라 기준금리가 1.50%로 0.25%포인트 오를 걸로 전망했다. 또한 NH투자증권도 국내외 경기 불확실성 등으로 한은이 연내 기준금리 인상에 나설 가능성이 작다고 봤다. 한편 하나금융투자는 정부의 가계부채 대책의 정책공조를 근거로 연내 인상할 가능성이 있다고 내다봤다."))

docs = ["[스페셜경제=유민주 기자]오는 31일 한국은행 금융통화위원회 본회의에서 기준 금리를 결정하기 때문에 관련 시장에서 관심이 고조되고 있다. 26일 금융권에 따르면 한국은행은 31일 8월 금융통화위원회 본 회의를 열고 현재 1.25%인 기준 금리의 변동 여부를 결정한다. 한은은 지난달 금통위에서 기준 금리를 1.25%수준에서 13개월째 동결했다. 앞서 이주열 총재가 지난달부터 꾸준히 금리인상 시그널을 보냈지만 시장은 이번달 금통위가 대내외 경제 변수를 좀 더 지켜보기 위해 금리를 동결할 것으로 관측이 우세하다. 우선 금융권은 최근 경기 회복세가 주춤하는 모습을 보여 좀 더 상황을 지켜보자는 판단을 내릴 것이란 예상하고 있다. 경제협력개발기구(OECD)에 따르면 지난 6월 한국의 경기선행지수(CLI)는 100.57이다. 전달(100.60) 대비 0.03포인트 하락했다. 이 지수는 6~9개월 뒤 경기흐름을 예측하는 지표로 3개월째 내림세다. 작년 말부터 이어진 경기 개선 흐름이 약해진 상황 가운데, OECD 경기선행지수도 하락세를 지속하면서 우리 경제의 불확실성이 다시 불거지는 모양새다. 또한 경기회복세를 이끌었던 수출 증가세 그 폭이 급격하게 둔화되고 있다. 한국은행이 발표한 7월 수출물량지수(139.42)는 전년동월대비 0.1% 올라 그 폭이 미미했다. 하지만 북한의 지정학적 리스크와 부동산 대책 등 경기 둔화 요인이 나타나고 있기 때문에 상황이 변했다. 최근 발표된 정부의 8 ·2 부동산대책 발표 이후 거래량이 대폭 줄면서, 주택경기 악화로 건설투자에 타격을 줄 것이란 우려가 커지고 있는 상태다. 올 2분기까지 건설투자의 성장기여도는 5분기 연속 50% 이상을 차지할 정도로 국내 경기에 큰 영향을 주고 있다. 다만 금융권은 한은 기준금리 인상 시점을 일단 내년으로 미뤄질 것으로 보고 있다. 국제금융센터에 따르면 해외 주요 투자은행 9곳 중 7곳은 내년 상반기 우리나라 기준금리가 1.50%로 0.25%포인트 오를 걸로 전망했다. 또한 NH투자증권도 국내외 경기 불확실성 등으로 한은이 연내 기준금리 인상에 나설 가능성이 작다고 봤다. 한편 하나금융투자는 정부의 가계부채 대책의 정책공조를 근거로 연내 인상할 가능성이 있다고 내다봤다."]


from collections import defaultdict
count = defaultdict(lambda:0)

for doc in docs :
    for word in doc.split():
        n = len(word)
        for e in range(1,n+1):
            count[word[:e]]+=1

word = "아이오아이는"
n = len(word)
count = defaultdict(lambda:0)

for e in range(2, n+1):
    w = word[:e]
    f = count[w]
    p = f/count[:e-1]
    print("{:6}, f={}, p={:.2}".format(w, f, s))

def cohesion(w):
    return pow(count[w]/count[w[0]],
               1/(len(w)-1))

word = "아이오아이가"
n = len(word)
count = defaultdict(lambda:0)

for e in range(2, n+1):
    w = word[:e]
    f = count[w]
    s = cohesion(w)
    print("{:6}, f={}, s={:.2}".format(w, f, s))

def ltokenize(w):
    n = len(w)
    if n <= 2: return (w,"")
    tokens = []
    for e in range(2, n+1):
        tokens.append(w[:e],w[e:],cohesion(w[:e]))
    tokens = sorted(tokens, Key=lambda x:-x[2])
    return tokens[0][:2]

sent = "뉴스의 기사를 이용했던 예시입니다"

for word in sent.split():
    print(ltokenize(word))

cohesion = {"파스":0.3, "파스타":0.7, "좋아요":0.2, "좋아":0.5}
score = lambda x: cohesion.get(x,0)