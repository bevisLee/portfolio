
## C101
def exp(x) :
    def factorial(n):
      fac = 1
      for i in range(1, n + 1):
        fac *= i
      return fac

    exp = 0.00000

    for i in range(0,n) :
        if i == 0 :
           a = 1/factorial(i)
        else :
           a = (float(x^i) / factorial(i))
        exp = exp + a

    return exp

n = 5
exp(3)

## C102
d = np.genfromtxt("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/haafor_test/Email Coding C102.txt", delimiter=',')[:,:-1]

where_are_NaNs = np.isnan(d)
d[where_are_NaNs] = 0

A = d[0]
B = d[1]
C = d[2]
D = d[3]
E = d[4]
F = d[5]
G = d[6]
H = d[7]
I = d[8]
J = d[9]

def C102(x) :
    mean = np.mean(x)
    median = np.median(x)
    print("mean :", mean)
    print("median :", median)

C102(C) # A ~ J insert

## C103
# a)
import numpy as np
import random

hexDec=[]
c = 1
with open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/haafor_test/hexDec.bin", "wb") as f: 
    for i in range(1,1000001):
        a = np.uint16(random.randrange(1, 100))
        hexDec.append(a)
        if i == (c*100000):
            np.array(hexDec).tofile(f)
            print (hexDec)
            print("--------------------")
            c = c + 1
            hexDec[:] = []

# b) 
import numpy as np
import random

hexDecSort=[]
with open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/haafor_test/hexDecSort.bin", "wb") as f: 
    for i in range(1,100001):
        a = np.uint16(random.randrange(1, 100))
        hexDecSort.append(a)
        if i == 100000 :
            hexDecSort.sort()
            np.array(hexDecSort).tofile(f)
            hexDecSort[:] = []

# c) 
import numpy as np
import random

with open("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/haafor_test/hexDecSizeSort.bin", "wb") as f: 
    for i in range(1,100001):
        hexDecSizeSort = np.fromfile('C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/haafor_test/hexDecSort.bin', dtype=np.uint16, count = i)
        from collections import Counter
        counts = Counter(hexDecSizeSort)
        if i == 100000 :
            print("hexDecSizeSort before :",hexDecSizeSort)
            hexDecSizeSort = sorted(hexDecSizeSort, key=lambda x: -counts[x])
            print("hexDecSizeSort after: ",hexDecSizeSort)
            np.array(hexDecSizeSort).tofile(f)
            hexDecSizeSort[:] = []

## M4. 단위 벡터(Unit Vector)
A) 45도 : 총 22개
1 : 점 0개
2 : 점 2개
3 : 점 0개
4 : 점 0개
5(적도) : 8개
6(적도 옆) : 6개
7(적도 반대쪽 옆) : 6개

B) 60도 : 16개
1 : 점 0개
2 : 점 2개
3 : 점 0개
4(적도) : 6개
5(적도 옆) : 4개
6(적도 반대쪽 옆) : 4개

## M5. 포커


## E1 할인된 현금흐름 문제
매년 임대료 = $10,000
1년 후 임대료 = $10,000    * 1.02 = $10,200
2년 후 임대료 = $10,200    * 1.02 = $10,404
3년 후 임대료 = $10,404    * 1.03 = $10,716.12
4년 후 임대료 = $10,716.12 * 1.03 = $11,037.6
5년 후 임대료 = $11,037.6  * 1.03 = $11,368.73
------------
전체 임대료 = $53,726.46

## P1. Regression
import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pandas.core import datetools
from scipy import stats

regression = pd.read_csv("C:/Users/bevis/Documents/Visual Studio 2017/Projects/Python_project/haafor_test/regression.csv",header = None)
regression.columns = ["A", "B", "C", "D"]

X = regression.loc[:,('A','B','C')]
y = regression.loc[:,'D']

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      D   R-squared:                       0.111
Model:                            OLS   Adj. R-squared:                  0.108
Method:                 Least Squares   F-statistic:                     41.45
Date:                Sun, 10 Dec 2017   Prob (F-statistic):           3.04e-25
Time:                        00:11:51   Log-Likelihood:                -5720.6
No. Observations:                1000   AIC:                         1.145e+04
Df Residuals:                     996   BIC:                         1.147e+04
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const        211.3946     83.877      2.520      0.012      46.798     375.991
A           -330.2889    121.219     -2.725      0.007    -568.163     -92.414
B            117.7610     38.141      3.087      0.002      42.914     192.608
C              1.5990      0.307      5.211      0.000       0.997       2.201
==============================================================================
Omnibus:                     1872.643   Durbin-Watson:                   2.005
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1884911.957
Skew:                          13.358   Prob(JB):                         0.00
Kurtosis:                     214.007   Cond. No.                         791.
==============================================================================
"""

# Prob (F-statistic):3.04e-25  -> 이 모델은 통계적으로 유의하다고 판단
# R-squared:0.111 -> 이 모델은 수정이 필요 / 보통 0.7은 되어야 함

# 각 변수는 p-value 0.05 이하, 변수의 갯수도 적어서 모델은 유의미하다고 판단하나, R-squared를 높이기 위해 데이터 파악

import matplotlib.pyplot as plt
regression.hist(bins=50, figsize=(20,15))
plt.show()

# C가 왼쪽으로 치우쳐서, 전체 변수 log 치환
regression_log = np.log(regression)
X_log = regression_log.loc[:,('A','B','C')]
y_log = regression_log.loc[:,'D']

regression_log.hist(bins=50, figsize=(20,15))
plt.show()

X3 = sm.add_constant(X_log)
reg = sm.OLS(y_log, X3)
reg_all = reg.fit()
print(reg_all.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      D   R-squared:                       0.625
Model:                            OLS   Adj. R-squared:                  0.624
Method:                 Least Squares   F-statistic:                     553.5
Date:                Sun, 10 Dec 2017   Prob (F-statistic):          1.32e-211
Time:                        00:14:30   Log-Likelihood:                -1499.3
No. Observations:                1000   AIC:                             3007.
Df Residuals:                     996   BIC:                             3026.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.1771      0.036      4.911      0.000       0.106       0.248
A             -0.4264      2.669     -0.160      0.873      -5.663       4.810
B              4.4367      0.923      4.805      0.000       2.625       6.249
C              0.0252      0.047      0.536      0.592      -0.067       0.118
==============================================================================
Omnibus:                      131.764   Durbin-Watson:                   2.026
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              703.410
Skew:                           0.469   Prob(JB):                    1.80e-153
Kurtosis:                       7.000   Cond. No.                         145.
==============================================================================
"""

# Prob (F-statistic):1.32e-211 -> 통계적으로 유의
# R-squared:0.625 -> 처음보다 높아져 모델이 쓸만해짐
# p-value "A" , "C"가 높게 나타나, "A", "C"를 제외하고 회귀 분석 진행

b = X_log.B

X4 = sm.add_constant(b)
reg = sm.OLS(y_log, X4)
reg_b = reg.fit()
print(reg_b.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      D   R-squared:                       0.625
Model:                            OLS   Adj. R-squared:                  0.625
Method:                 Least Squares   F-statistic:                     1663.
Date:                Sun, 10 Dec 2017   Prob (F-statistic):          9.27e-215
Time:                        00:16:15   Log-Likelihood:                -1499.5
No. Observations:                1000   AIC:                             3003.
Df Residuals:                     998   BIC:                             3013.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.1760      0.036      4.892      0.000       0.105       0.247
B              4.4129      0.108     40.781      0.000       4.201       4.625
==============================================================================
Omnibus:                      131.563   Durbin-Watson:                   2.025
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              702.576
Skew:                           0.468   Prob(JB):                    2.74e-153
Kurtosis:                       6.998   Cond. No.                         3.19
==============================================================================
"""
#  Prob (F-statistic):9.27e-215 -> 통계적으로 유의
# R-squared:0.625 -> 처음보다 높아져 모델이 쓸만해짐
# p-value 도 0.05보다 낮게 나타나, 적합한 모델로 판단
