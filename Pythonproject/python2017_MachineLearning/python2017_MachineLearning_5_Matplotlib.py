
### Matplotlib
# 파이썬에서 그래프를 그릴 때 주로 사용하는 플로팅 라이브러리인 matplotlib에 대해 간단하게 공부해 봅시다. 이 강의는 Nicolas P. Rougier의 Matplotlib tutorial 을 번역하고 실습 플랫폼에 연결한 것입니다.

# 소개
# matplotlib는 2D 그래픽 용으로 가장 많이 사용되는 Python 패키지일 것입니다. Python의 데이터를 시각화하는 매우 빠른 방법과 다양한 형식의 출판 품질 스케일을 제공합니다. 가장 일반적인 경우를 다루는 대화형 모드에서 matplotlib를 배워봅시다.

# IPython과 pylab 모드
# IPython은 이름이 지정된 입출력, 셸 명령에 대한 액세스, 개선 된 디버깅 등을 비롯한 많은 흥미로운 기능을 갖춘 향상된 대화형 Python 셸입니다. 명령 줄 인수 -pylab (IPython 버전 0.12 이후 --pylab)으로 시작하면 Matlab / Mathematica와 유사한 기능을 가진 대화식 matplotlib 세션을 사용할 수 있습니다.

# pyplot
# pyplot은 matplotlib 객체 지향 플로팅 라이브러리에 편리한 인터페이스를 제공합니다. pyplot은 MATLAB(TM)을 면밀히 모델링했습니다. 따라서, pyplot에서 플로팅 명령의 대부분은 유사한 인수를 가진 MATLAB(TM) 명령을 사용합니다. 중요한 명령은 대화형 예제로 설명하겠습니다.

## 단순한 플롯(plot)
# 이 섹션에서는 같은 플롯에 코사인 함수와 사인 함수를 그립니다. 기본 설정에서 시작하여 그림을 단계적으로 풍부하게 만들어 더 멋지게 만듭니다.

# 첫 번째 단계는 사인 및 코사인 함수 데이터를 얻는 것입니다.

import numpy as np
from IPython.display import display

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
display(np.array([C.T, S.T]))

# X는 이제 -π에서 + π까지 (포함 된) 256개의 값을 갖는 배열입니다. C는 코사인 (256개)이고 S는 사인 (256개)입니다.

## 기본값 사용
# Matplotlib에는 모든 종류의 속성을 사용자 정의 할 수 있는 일련의 기본 설정이 있습니다. matplotlib의 거의 모든 속성의 기본값들인 그림 크기 및 dpi, 선 너비, 색상 및 스타일, 축, 축 및 격자 속성, 텍스트 및 글꼴 속성 등을 제어할 수 있습니다. 대부분의 경우 matplotlib의 기본값은 좋습니다만 특정 경우 일부 속성을 수정해야 할 수 있습니다.

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

plt.plot(X, C)
plt.plot(X, S)

plt.show()

# 아래 스크립트에서 플롯의 모양에 영향을 미치는 모든 그림 설정을 인스턴스화 (및 주석 처리)했습니다. 설정을 명시적으로 기본값으로 설정했지만, 바로 값을 수정하고 어떤 영향이 있는지 확인할 수 있습니다 (아래의 선 속성 및 선 스타일 참조).

# Imports
import numpy as np
import matplotlib.pyplot as plt

# 8x6 포인트, 인치당 80도트의 새 figure를 만든다.
plt.close()
plt.figure(figsize=(8,6), dpi=80)

# 1x1 그리드의 서브플롯을 만든다.
plt.subplot(111)

X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
C,S = np.cos(X), np.sin(X)

# 코사인 값을 1픽셀의 파란 연속선으로 그린다.
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# 사인 값을 1픽셀의 초록 연속선으로 그린다.
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")

# x축 한계를 정한다.
plt.xlim(-4.0, 4.0)

# x축 틱을 정한다.
plt.xticks(np.linspace(-4,4,9,endpoint=True))

# y축 한계를 정한다.
plt.ylim(-1.0, 1.0)

# y축 틱을 정한다.
plt.yticks(np.linspace(-1,1,5,endpoint=True))

# 인치당 72도트의 그림을 저장한다.
# savefig("../figures/exercice_2.png",dpi=72)

# 결과를 화면에서 확인한다.
plt.show()

## 색상 및 선폭 변경
# 첫 번째 단계에서는 코사인을 파란색으로, 사인을 빨간색으로, 두 번째로 약간 더 두꺼운 선을 그려봅시다. 좀 더 가로로 길게 만들기 위해 그림 크기를 약간 변경합니다.

plt.close()
plt.figure(figsize=(10,6), dpi=80)
plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-")
plt.show()

## 한계 설정
# 그림의 현재 한계는 너무 빡빡하므로, 모든 데이터 포인트를 명확하게 보기 위해 약간 공간을 만들어 봅시다.

plt.xlim(X.min()*1.1, X.max()*1.1)
plt.ylim(C.min()*1.1, C.max()*1.1)
plt.show()

# 틱 설정
# 현재 틱은 사인 및 코사인에 흥미로운 값 (+/-π, +/-π/2)을 표시하지 않기 때문에 이상적이지 않습니다. 이 값만 표시되도록 변경합니다.

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks([-1, 0, +1])
plt.show()

# 틱 레이블 설정하기
# 틱은 이제 올바르게 배치되지만 레이블은 그다지 명확하지 않습니다. 우리는 3.142가 π임을 추측 할 수 있지만 그것을 명확하게하는 것이 낫습니다. 눈금 값을 설정할 때 두 번째 인수 목록에 해당 레이블을 제공 할 수도 있습니다. 라벨의 멋진 렌더링을 위해 LaTeX를 사용합니다.

plt.xticks([- np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.yticks([- 1, 0, +1],
        [r'$-1$', r'$0$', r'$+1$'])
plt.show()

## 스파인(spine) 옮기기
# 스파인(spine)은 축 눈금을 연결하고 데이터 영역의 경계를 나타내는 선입니다. 임의의 위치에 배치할 수 있는데, 지금까지는 축의 경계에 있었습니다. 중간으로 옮기고 싶으므로 바꾸어 봅시다. 4개 (상단 / 하단 / 좌측 / 우측)가 있으므로, 색상을 none으로 설정하여 상단과 우측을 버리고 하단과 좌측을 데이터 공간 좌표에서 0으로 조정합니다.

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.show()

# 범례 추가하기
# 왼쪽 상단 모서리에 범례를 추가합시다. 이는 플롯 명령에 키워드 인수 레이블 (범례 상자에서 사용됨)을 추가하기 만하면됩니다.

plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
plt.plot(X, S, color="red",  linewidth=2.5, linestyle="-", label="sine")

plt.legend(loc='upper left', frameon=False)
plt.show()

## 몇 가지 주석 달기
# annotate 명령을 사용하여 흥미로운 점에 주석을 달아 봅시다. 2π/3 값을 선택해서 사인과 코사인 모두에 주석을 달아봅시다.처음에는 직선의 점선뿐만 아니라 곡선에 마커를 그립니다. 그런 다음 annotate 명령을 사용하여 화살표가 있는 텍스트를 표시합니다. 

t = 2*np.pi/3
plt.plot([t,t], [0,np.cos(t)], color ='blue', linewidth=1.5, linestyle="--")
plt.scatter([t,], [np.cos(t),], 50, color ='blue')

plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.plot([t,t], [0,np.sin(t)], color ='red', linewidth=1.5, linestyle="--")
plt.scatter([t,], [np.sin(t),], 50, color ='red')

plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
             xy=(t, np.cos(t)), xycoords='data',
             xytext=(-90, -50), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
plt.show()

## 악마는 디테일에 있다
# 틱 레이블은 파란색과 빨간색 선으로 인해 거의 보이지 않습니다. 레이블을 더 크게 만들 수 있고 반투명 한 흰색 바탕에 그 속성들이 표현되도록 속성을 조정할 수 있습니다. 이렇게하면 데이터와 라벨을 모두 볼 수 있습니다.

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))
plt.show()

## figures, subplots, axes 및 ticks
# 지금까지는 암묵적으로 정해져 있는 그림과 축 생성을 사용했습니다. 이는 빠르게 그래프를 그릴 때 유용합니다. figure, subplot 및 axes를 명시적으로 사용하여 디스플레이를 보다 잘 제어 할 수 있습니다. matplotlib의 figure는 사용자 인터페이스의 전체 창을 의미합니다. 이 figure에는 하위 그림이 있을 수 있습니다. subplot은 플롯을 일반 그리드에 배치하지만 축은 그림 내에서 자유 배치를 허용합니다. 둘 다 의도에 따라 유용할 수 있습니다. 우리는 이미 명시적으로 호출하지 않고도 그림 및 하위 그림으로 작업해 보았습니다. plot을 호출하면 matplotlib은 gca()를 호출하여 현재 축을 가져오고, gca는 차례대로 gcf()를 호출하여 현재 그림을 가져옵니다. 아무것도 없으면 figure()를 호출하여 엄밀히 말하면 subplot(111)을 만듭니다. 세부 사항을 살펴 봅시다.

# 피겨 (figure)
# 피겨는 GUI에서 "figure #" 제목 인 창입니다. 숫자는 0부터 시작하는 일반적인 파이썬 방법과는 다르게 1부터 시작하여 번호가 매겨집니다. 분명히 MATLAB 스타일입니다. 그림의 모양을 결정하는 몇 가지 매개 변수가 있습니다.

# 기본값은 리소스 파일에 지정 될 수 있으며 대부분의 시간동안 사용됩니다. figure의 숫자만 자주 변경됩니다.

# GUI로 작업 할 때 오른쪽 상단의 x를 클릭하여 그림을 닫을 수 있습니다. 그러나 close를 호출하여 프로그래밍 방식으로 도형을 닫을 수 있습니다. 인수에 따라 (1) 현재 숫자 (인수 없음), (2) 특정 숫자 (그림 숫자 또는 인수 인스턴스) 또는 (3) 모든 숫자 (모두 인수로)를 닫습니다.

# 다른 객체와 마찬가지로 set_something 메소드를 사용하여 Figure 속성을 설정할 수 있습니다.

## 서브 플롯 (subplot)
# 서브 플롯을 사용하면 일반 격자에 플롯을 정렬 할 수 있습니다. 행과 열의 수와 플롯의 수를 지정해야합니다. gridspec 명령은 더 강력한 대안입니다.

# 축 (axes)
# 축은 서브 플롯과 매우 유사하지만 피겨의 어느 위치 에나 그림을 배치 할 수 있습니다. 작은 그림을 큰 그림 안에 넣고 싶다면 축을 사용하십시오.

# 틱 로케이터
# 서로 다른 종류의 요구에 맞는 여러 로케이터가 있습니다.

# 이 모든 로케이터는 기본 클래스 matplotlib.ticker.Locator에서 파생됩니다. 자신만의 로케이터도 파생시킬 수 있습니다. 틱으로 날짜를 처리하는 것은 특히 까다롭습니다. 그래서 matplotlib은 matplotlib.dates에 특별한 로케이터를 제공합니다.

## 애니매이션 
# 오랜 시간 동안 matplotlib의 애니메이션은 쉬운 작업이 아니며 주로 영리한 해킹을 통해 수행되었습니다. 그러나 버전 1.1부터 매우 직관적으로 애니메이션을 제작할 수있는 도구가 등장하여 모든 종류의 형식으로 저장할 수있게 되었습니다 (하지만 60fps에서 매우 복잡한 애니메이션을 실행할 수 있을 것으로 기대하지는 않습니다).

# matplotlib에서 애니메이션을 만드는 가장 쉬운 방법은 matplotlib에 업데이트 할 그림, 업데이트 함수 및 프레임 사이의 지연을 지정하는 FuncAnimation 객체를 선언하는 것입니다.

## 비내리기
# 매우 단순한 비 효과는 그림 위에 작은 성장 고리를 무작위로 배치함으로써 얻을 수 있습니다. 물론, 파도가 시간이 지남에 따라 가라않기 때문에 그들은 영원히 성장하지 않을 것입니다. 비 효과를 시뮬레이트하기 위해 링이 성장함에 따라 더 이상 보이지 않는 지점까지 점점 더 투명한 색을 사용할 수 있습니다. 이 시점에서 링을 제거하고 새 링을 만듭니다.

# 첫 번째 단계는 빈 그림을 만드는 것입니다.

plt.close()
# New figure with white background
fig = plt.figure(figsize=(6,6), facecolor='white')

# New axis over the whole figure, no frame and a 1:1 aspect ratio
ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)

# 다음으로 여러 개의 링을 만들어야합니다. 이를 위해 우리는 일반적으로 포인트들을 시각화하는 데 사용하는 스캐터 플롯 (산점도) 객체를 사용할 수 있지만, facecolor 가 없도록 지정해서 링을 그릴 수도 있습니다. 또한 최소 크기와 최대 크기 사이의 모든 크기가 있으면서 가장 큰 링은 거의 투명하도록 각 링의 초기 크기와 색상을 지정해야합니다.

# 링 갯수
n = 50
size_min = 50
size_max = 50*50

# 링 위치
P = np.random.uniform(0,1,(n,2))

# 링 색상
C = np.ones((n,4)) * (0,0,0,1)
# Alpha color channel goes from 0 (transparent) to 1 (opaque)
C[:,3] = np.linspace(0,1,n)

# 링 크기
S = np.linspace(size_min, size_max, n)

# 스캐터 플롯
scat = ax.scatter(P[:,0], P[:,1], s=S, lw = 0.5,
                  edgecolors = C, facecolors='None')

# [0,1] 로 축 한계를 지정하고 틱을 없애기
ax.set_xlim(0,1), ax.set_xticks([])
ax.set_ylim(0,1), ax.set_yticks([])
plt.show()

# 이제 애니메이션을 위한 업데이트 함수를 작성해야합니다. 각 시간 간격마다 각 고리가 더 투명해져야하며 큰 고리는 완전히 투명해지고 없어져야 합니다. 물론, 실제로 가장 큰 링을 제거하는 대신 새로운 임의의 위치에서 아주 작은 크기와 색상의 새 링으로 재사용합니다. 따라서 링의 수가 일정하게 유지됩니다.

def update(frame):
    global P, C, S

    # 모든 링들이 조금 더 투명해짐
    C[:,3] = np.maximum(0, C[:,3] - 1.0/n)

    # 각 링이 조금 커짐
    S += (size_max - size_min) / n

    # 프레임 번호에 따라 링을 리셋함
    i = frame % 50
    P[i] = np.random.uniform(0,1,2)
    S[i] = size_min
    C[i,3] = 1

    # 스캐터 객체 업데이트
    scat.set_edgecolors(C)
    scat.set_sizes(S)
    scat.set_offsets(P)

    # 수정한 오브젝트를 리턴한다
    return scat,

# 마지막 단계는 이 함수를 애니메이션의 업데이트 함수로 사용하여 결과를 표시하거나 동영상으로 저장하도록 matplotlib에 지시하는 것입니다.

from matplotlib.animation import FuncAnimation # FuncAnimation import

animation = FuncAnimation(fig, update, interval=10, blit=True, frames=200)
# animation.save('rain.gif', writer='imagemagick', fps=30, dpi=40)
plt.show()

## 지진
# 비 애니메이션을 사용하여 지난 30일 동안 지구에서 일어난 지진을 시각화합니다. USGS 지진 재해 프로그램은 국가 지진 재해 감소 프로그램 (NEHRP)의 일환으로 웹사이트에서 여러 데이터를 제공합니다. 이러한 데이터는 지진의 크기에 따라 정렬되며, 주요 지진부터 작은 지진까지 모두 담고 있습니다. 지구에서 매 시간마다 일어나는 경미한 지진의 횟수에 놀랄 것입니다. 이것이 너무 많은 데이터이므로, 우리는 > 4.5 크기의 지진만을 사용하겠습니다. 글을 쓰는 시점에서, 이미 지난 30 일 동안 300 건 이상의 지진을 표시하고 있습니다.

# 첫 번째 단계는 데이터를 읽고 변환하는 것입니다. 우리는 원격 데이터를 열고 읽을 수있는 urllib 라이브러리를 사용할 것입니다. 웹 사이트의 데이터는 첫 번째 줄에 내용이 제공되는 CSV 형식을 사용합니다.

# data 참고
# time,latitude,longitude,depth,mag,magType,nst,gap,dmin,rms,net,id,updated,place,type
# 2015-08-17T13:49:17.320Z,37.8365,-122.2321667,4.82,4.01,mw,...
# 2015-08-15T07:47:06.640Z,-10.9045,163.8766,6.35,6.6,mwp,...

# 우리는 위도, 경도 및 크기에만 관심이 있으며 사건의 시간은 분석하지 않겠습니다. (네. 아쉽죠. 풀 리퀘스트를 보내주세요.)

import urllib
# from mpl_toolkits.basemap import Basemap #basemap install 에러
import geocoder

# -> http://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php
feed = "http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/"

# Significant earthquakes in the last 30 days
# url = urllib.request.urlopen(feed + "significant_month.csv")

# Magnitude > 4.5
url = urllib.request.urlopen(feed + "4.5_month.csv")

# Magnitude > 2.5
# url = urllib.request.urlopen(feed + "2.5_month.csv")

# Magnitude > 1.0
# url = urllib.request.urlopen(feed + "1.0_month.csv")

# Reading and storage of data
data = url.read()
data = data.split(b'\n')[+1:-1]
E = np.zeros(len(data), dtype=[('position',  float, 2),
                               ('magnitude', float, 1)])

for i in range(len(data)):
    row = data[i].split(',')
    E['position'][i] = float(row[2]),float(row[1])
    E['magnitude'][i] = float(row[4])

# 이제 지진의 중심을 정확히 보여주고 matplotlib이 처리 할 수있는 좌표의 위도 / 경도를 번역하기 위해 그림에 지구를 그려야합니다. 다행스럽게도, 설치 및 사용이 매우 간편한 기본 맵 프로젝트 (보다 완벽한 cartopy 로 대체되는 경향이 있음)가 있습니다. 첫 번째 단계는 스크린에 지구를 그리는 투영법을 정의하는 것입니다 (많은 다른 투영법이 존재합니다). 그리고 저는 우리 같은 비 전문가를위한 표준인 밀 프로젝션을 사용할 것입니다.
fig = plt.figure(figsize=(14,10))
ax = plt.subplot(1,1,1)

earth = Basemap(projection='mill')

# 다음으로 우리는 해안선을 그리고 대륙을 채우겠습니다.

earth.drawcoastlines(color='0.50', linewidth=0.25)
earth.fillcontinents(color='0.95')

# 지구 객체는 좌표를 자동으로 변환하는 데에도 사용됩니다. 거의 끝났습니다. 마지막 단계는 비 코드를 적용하고 눈요기를 하는겁니다.

P = np.zeros(50, dtype=[('position', float, 2),
                         ('size',     float, 1),
                         ('growth',   float, 1),
                         ('color',    float, 4)])
scat = ax.scatter(P['position'][:,0], P['position'][:,1], P['size'], lw=0.5,
                  edgecolors = P['color'], facecolors='None', zorder=10)

def update(frame):
    current = frame % len(E)
    i = frame % len(P)

    P['color'][:,3] = np.maximum(0, P['color'][:,3] - 1.0/len(P))
    P['size'] += P['growth']

    magnitude = E['magnitude'][current]
    P['position'][i] = earth(*E['position'][current])
    P['size'][i] = 5
    P['growth'][i]= np.exp(magnitude) * 0.1

    if magnitude < 6:
        P['color'][i]    = 0,0,1,1
    else:
        P['color'][i]    = 1,0,0,1
    scat.set_edgecolors(P['color'])
    scat.set_facecolors(P['color']*(1,1,1,0.25))
    scat.set_sizes(P['size'])
    scat.set_offsets(P['position'])
    return scat,

animation = FuncAnimation(fig, update, interval=10)
plt.show()

## 다른 유형의 플롯
# 일반 플롯

import numpy as np
import matplotlib.pyplot as plt
plt.close()

n = 256
X = np.linspace(-np.pi,np.pi,n,endpoint=True)
Y = np.sin(2*X)

plt.plot (X, Y+1, color='blue', alpha=1.00)
plt.plot (X, Y-1, color='blue', alpha=1.00)
plt.show()

# 스캐터 플롯 (산포도)

import numpy as np
import matplotlib.pyplot as plt
plt.close()

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)

plt.scatter(X,Y)
plt.show()

# 바 플롯

import numpy as np
import matplotlib.pyplot as plt
plt.close()

n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

plt.ylim(-1.25,+1.25)
plt.show()

# 칸토어 플롯

import numpy as np
import matplotlib.pyplot as plt
plt.close()

def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.show()

# Imshow

import numpy as np
import matplotlib.pyplot as plt
plt.close()

def f(x,y): return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 10
x = np.linspace(-3,3,4*n)
y = np.linspace(-3,3,3*n)
X,Y = np.meshgrid(x,y)

plt.imshow(f(X,Y))
plt.show()

# 파이 차트

import numpy as np
import matplotlib.pyplot as plt

plt.close()

n = 20
Z = np.random.uniform(0,1,n)
plt.pie(Z)
plt.show()

# 쿼버 플롯

import numpy as np
import matplotlib.pyplot as plt
plt.close()

n = 8
X,Y = np.mgrid[0:n,0:n]
plt.quiver(X,Y)
plt.show()

# 그리드

import numpy as np
import matplotlib.pyplot as plt
plt.close()

axes = gca()
axes.set_xlim(0,4)
axes.set_ylim(0,3)
axes.set_xticklabels([])
axes.set_yticklabels([])

plt.show()

# 멀티 플롯

import numpy as np
import matplotlib.pyplot as plt
plt.close()

plt.subplot(2,2,1)
plt.subplot(2,2,3)
plt.subplot(2,2,4)

plt.show()

# 국축

import numpy as np
import matplotlib.pyplot as plt
plt.close()

plt.axes([0,0,1,1])

N = 20
theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
radii = 10*np.random.rand(N)
width = np.pi/4*np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor( cm.jet(r/10.))
    bar.set_alpha(0.5)

plt.show()

# 3D 플롯

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close()

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

plt.show()

# 텍스르

