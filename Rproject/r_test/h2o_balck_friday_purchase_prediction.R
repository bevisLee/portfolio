# 소매 회사 "ABC Private Limited"는 여러 카테고리의 다양한 제품에 대한 
# 고객 구매 행동 (구체적으로 구매 금액)을 이해하고자합니다. 

# 그들은 지난 달의 엄선 된 대량 제품에 대한 다양한 고객의 구매 요약을 공유했습니다.

# 데이터 세트에는 지난 달의 고객 인구 통계(연령, 성별, 결혼 상태, 도시 _ 유형, 숙박 _ 현재 _city), 
# 제품 세부 정보(product_id 및 제품 카테고리) 및 총 구매 _ 금액도 포함됩니다.

# 이제 그들은 다양한 제품에 대한 고객의 구매 금액을 예측하는 모델을 구축하여 
# 서로 다른 제품에 대한 고객을위한 맞춤형 제안을 만드는 데 도움이됩니다.

# data definition
# User_ID                    : User ID
# Product_ID                 : Product ID
# Gender                     : Sex of User
# Age                        : Age in bins
# Occupation                 : Occupation(Masked) 직업
# City_Category              : Category of the City(A, B, C)
# Stay_In_Current_City_Years : Number of years stay in current city
# Marital_Status             : Marital Status 결혼 여부
# Product_Category_1         : Product Category(Masked)
# Product_Category_2         : Product may belongs to other category also(Masked)
# Product_Category_3         : Product may belongs to other category also(Masked)
# Purchase Purchase          : Amount(Target Variable)

if (!require("data.table")) { install.packages("data.table") }

####_1. Data load and first Prediction export -----------------------
# load data using fread
train <- fread("C:/Users/bevis/Downloads/Black_Friday/train.csv", stringsAsFactors = T)
test1 <- fread("C:/Users/bevis/Downloads/Black_Friday/test.csv", stringsAsFactors = T)
test <- test1[,-12]

#No. of rows and columns in Train, test
dim(train)
dim(test)

#Check NA data
colSums(sapply(train, is.na)) # Product_Category_2 : 173,638 / Product_Category_3 : 383,247 

# 총 12개 변수 중 2개가 많은 NA 포함
# 구매는 종속 변수, 나머지는 독립 변수
# 구매 변수(연속적)의 본질을 살펴보면, 이것은 회귀 문제라고 추론할 수 있음
# first prediction 은 train data의 평균 결제액으로 베이스라인 설정

#first prediction using mean
sub_mean <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = mean(train$Purchase))
write.csv(sub_mean, file = "C:/Users/bevis/Downloads/Black_Friday/first_sub.csv", row.names = F)

#Check train and test
str(train) # 12 variables
str(test) # 11 variables

summary(train)
summary(test)
# Product_Category_1, 2, 3 의 최대값 불일치 / 카테고리 레벨의 잡음 제거 필요

#combine data set
test[, Purchase := mean(train$Purchase)]
c <- list(train, test)
combin <- rbindlist(c) # rbind 보다 빠름

####_2. EDA -----------------------
## 변수 간의 관계 이해 : 단변량 분석 vs 이변량 분석
# 단변량(univariate analysis) : 종속변수가 1개인 경우 / T-test, ANOVA, 회귀분석 활용
    # T-test : 두 개의 집단의 평균 차이가 유의한지 분석
    # ANOVA : 세 개 이상 집단의 평균 차이가 유의한지 분석
 # 이변량(bivariate analysis) : 종속변수 1개, 독립변수 2개 / 상관분석 활용
    # 두 개의 변수만이 포함되는 통계적 분석이다. 
    # 두 변수간의 독립성이나 관련성을 알아보기 위한 빈도분포 분석이나, 
    # 단순상관이나 회귀분석, 그리고 하나의 독립변수와 하나의 종속변수로 
    # 이루어지는 평균의 차이검정 등은 이변량분석의 예들이다.
 # 다변량 : 종속변수 2개이상 / 요인분석, 군집분석, 정준상관분석, 다차원척도법, 등 활용

#analyzing gender variable
combin[, prop.table(table(Gender))] # Gender 비율 확인
 # -> F = 1, M = 0 인코딩 필요

#Age Variable
combin[, prop.table(table(Age))] # Age 비율 확인

#City Category Variable
combin[, prop.table(table(City_Category))] # City 비율 확인

#Stay in Current Years Variable
combin[, prop.table(table(Stay_In_Current_City_Years))] # 거주년수 비율 확인

#unique values in ID variables
length(unique(combin$Product_ID)) # Product ID 수 확인

length(unique(combin$User_ID)) # User ID 수 확인

#missing values
colSums(is.na(combin))

# 단변량 분석을 통한 추론
 # Gender 변수를 0,1로 인코딩
 # Age를 다시 코딩 해야함
 # City_Category 변수 3개를 인코딩
 # Stay_In_Current_City_Years 의 "4+" 수준을 재평가 해야 함
 # 2개의 변수만 NA가 존재(누락된 값), 숨겨진 추세는 누락된 값에 포함된 경우가 많음
 # 데이터 세트에 모든 고유 ID가 포함되어 있지 않아, 피쳐 엔지니어링 필요함
# 데이터 랭글러(data wrangler) : 대량의 데이터를 거르고, 변환하고, 합치고, 분류하는 전문가

## 시각화를 통한 변수 파악
if (!require("ggplot2")) { install.packages("ggplot2") }

## 참조(plot를 통한 분석) - https://www.analyticsvidhya.com/blog/2016/03/questions-ggplot2-package-r/

# 막대 그래프 : 범주형 변수 또는 연속 및 범주형 변수의 조합을 그릴 때 사용
#Age vs Gender
ggplot(combin, aes(Age, fill = Gender)) + geom_bar()

#Age vs City_Category
ggplot(combin, aes(Age, fill = City_Category)) + geom_bar()

## 범주형 변수를 분석하기 위한 교차 테이블 생성
if (!require("gmodels")) { install.packages("gmodels") }
CrossTable(combin$Occupation, combin$City_Category)
 
####_2-1. T-test & ANOVA ------------------------
## Gender 기준, T-test
t.test(Purchase ~ Gender, data = combin,
    alternative = c("two.sided"), # c("two.sided : 같은지", "less : 작은지", "greater : 큰지")
    var.equal = TRUE, conf.level = 0.95)

# data:Purchase by Gender
# t = -44.802, df = 783660, p - value < 2.2e-16
# alternative hypothesis:true difference in means is not equal to 0
# 95 percent confidence interval:
# -514.7318 - 471.5829
# sample estimates:
# mean in group F mean in group M
# 8892.665 9385.823

# --> p - value < 2.2e-16 = .00000000000000022 / 2.2의 -16배수

## Purchase 기준, 평균 구매액 9263.969 T-test
t.test(combin$Purchase,
    alternative = "greater",
    mu = 9263.969, con.level = 0.95)

# data:combin$Purchase
# t = -6.0381e-05, df = 783670, p - value = 0.5
# alternative hypothesis:true mean is greater than 9263.969
# 95 percent confidence interval:
# 9256.149 Inf
# sample estimates:
# mean of x
# 9263.969

## Age 기준, ANOVA
bartlett.test(Purchase ~ Age, data = combin)

# Bartlett test of homogeneity of variances

# data:Purchase by Age
# Bartlett 's K-squared = 68.914, df = 6, p-value = 6.827e-13

## City_Category 기준, ANOVA
bartlett.test(Purchase ~ City_Category, data = combin)

# Bartlett test of homogeneity of variances

# data:Purchase by City_Category
# Bartlett 's K-squared = 967.88, df = 2, p-value < 2.2e-16

####_3. Data wrangling use data.table ------------------------
## NA가 존재하는 변수를 새로운 변수로 만들고 기존 변수를 평가
# Product_Category_2 --> Product_Category_2_NA
# Product_Category_3 --> Product_Category_3_NA

#create a new variable for missing values
combin[, Product_Category_2_NA := ifelse(sapply(combin$Product_Category_2, is.na) == TRUE, 1, 0)]
combin[, Product_Category_3_NA := ifelse(sapply(combin$Product_Category_3, is.na) == TRUE, 1, 0)]

#impute missing values : NA에 임의으 숫자로 대체
combin[, Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999", Product_Category_2)]
combin[, Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999", Product_Category_3)]

#set column level

# Stay_In_Current_City_Years 의 "4+" 수준을 재평가
levels(combin$Stay_In_Current_City_Years)[levels(combin$Stay_In_Current_City_Years) == "4+"] <- "4"

#recoding age groups
levels(combin$Age)[levels(combin$Age) == "0-17"] <- 0
levels(combin$Age)[levels(combin$Age) == "18-25"] <- 1
levels(combin$Age)[levels(combin$Age) == "26-35"] <- 2
levels(combin$Age)[levels(combin$Age) == "36-45"] <- 3
levels(combin$Age)[levels(combin$Age) == "46-50"] <- 4
levels(combin$Age)[levels(combin$Age) == "51-55"] <- 5
levels(combin$Age)[levels(combin$Age) == "55+"] <- 6

## 모델링 목적을 위해 요인 변수를 숫자 또는 정수로 변환
#convert age to numeric
combin$Age <- as.numeric(combin$Age)

#convert Gender into numeric
combin[, Gender := as.numeric(as.factor(Gender)) - 1]

#User Count : 사용자별 구매수 파악
combin[, User_Count := .N, by = User_ID]

#Product Count : 인기 상품 파악
combin[, Product_Count := .N, by = Product_ID]

#Mean Purchase of Product : 제품별 제품의 평균 구매 가격(구매 가격을 낮추면 구매할 확률이 높아지거나 그 반대가 될 수 있음)
combin[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]

#Mean Purchase of User : 사용자별로 평균 구매 가격을 매핑하는 또 다른 변수, 즉 사용자가 평균 구매 한 금액을 매핑
combin[, Mean_Purchase_User := mean(Purchase), by = User_ID]

## City_Category 3개 인코딩
if (!require("dummies")) { install.packages("dummies") }
combin <- dummy.data.frame(combin, names = c("City_Category"), sep = "_") # City_Category_A, B, C 분리 1 / 0 으로 구분

#check classes of all variables : 변수 타입 확인
sapply(combin, class)

#converting Product Category 2 & 3
combin$Product_Category_2 <- as.integer(combin$Product_Category_2)
combin$Product_Category_3 <- as.integer(combin$Product_Category_3)

####_4-1. H2O use Modeling ------------------------
 # Regression: Starters Guide to Regression - https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/
 # Random Forest, GBM:Starters Guide to Tree Based Algorithms - https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
 # Deep Learning:Starters Guide to Deep Learning - https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/

#Divide into train and test
c.train <- combin[1:nrow(train),] # 550,068
c.test <- combin[-(1:nrow(train)),]

# train data Noise 제거를 위해 Product_Category_1 ~ 18까지의 모든 행을 선택하여 
# 카테고리 레벨 19 & 20의 행을 삭제하여 제거
c.train <- c.train[c.train$Product_Category_1 <= 18,] # 기존 : 550,068 --> 적용 : 545,915

if (!require("h2o")) { install.packages("h2o") }

# h2o 클러스터 시작
localH2O <- h2o.init(nthreads = -1)

# h2o 인스턴트 상태 확인
h2o.init()

#data to h2o cluster : 인스턴트로 데이터 전송
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)

#check column index number
colnames(train.h2o)

#dependent variable (Purchase)
y.dep <- 14

#independent variables (dropping ID variables)
x.indep <- c(3:13, 15:20)

####_4-2. Multiple Regression in H2O ------------------------
regression.model <- h2o.glm(y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian")

h2o.performance(regression.model)
# H2ORegressionMetrics:glm
# ** Reported on training data. **

# MSE:16710559
# R2:0.3261545
# Mean Residual Deviance:16710559
# Null Deviance:1.353804e+13
# Null D.o.F.:545914
# Residual Deviance:9.122545e+12
# Residual D.o.F.:545898
# AIC:10628689

#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.reg$predict)

write.csv(sub_reg, file = "C:/Users/bevis/Downloads/Black_Friday/sub_reg.csv", row.names = F)

####_4-3. Random Forest in H2O ------------------------
#Random Forest
system.time(
rforest.model <- h2o.randomForest(y = y.dep, x = x.indep, training_frame = train.h2o, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
)
# 사용자 시스템 elapsed
# 3.70   0.25   435.67

h2o.performance(rforest.model)
# H2ORegressionMetrics:drf
# ** Reported on training data. **
# Description:Metrics reported on Out - Of - Bag training samples

# MSE:10283633
# R2:0.5853173
# Mean Residual Deviance:10283633

#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))
# 사용자 시스템 elapsed
# 1.05   0.03   31.95

#writing submission file
sub_rf <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.rforest$predict)
write.csv(sub_rf, file = "C:/Users/bevis/Downloads/Black_Friday/sub_rf.csv", row.names = F)

####_4-4. GBM in H2O ------------------------
#GBM
system.time(
gbm.model <- h2o.gbm(y = y.dep, ## target: using the logged variable created earlier
                     x = x.indep, ## this can be names or column numbers
                     training_frame = train.h2o, ## H2O frame holding the training data
                     ntrees = 1000, ## use fewer trees than default (50) to speed up training
                     max_depth = 4, 
                     learn_rate = 0.01, ## lower learn_rate is better, but use high rate to offset few trees
                     seed = 1122)
)
# 사용자 시스템 elapsed
# 8.47   0893   935.09

h2o.performance(gbm.model)
# H2ORegressionMetrics:gbm
# ** Reported on training data. **

# MSE:6319672
# R2:0.7451622
# Mean Residual Deviance:6319672

#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.gbm$predict)
write.csv(sub_gbm, file = "C:/Users/bevis/Downloads/Black_Friday/sub_gbm.csv", row.names = F)

####_4-5. Deep-Learning in H2O ------------------------
#deep learning models
system.time(
             dlearning.model <- h2o.deeplearning(y = y.dep,
             x = x.indep,
             training_frame = train.h2o,
             epoch = 60,
             hidden = c(100, 100),
             activation = "Rectifier",
             seed = 1122
             )
)
# 사용자 시스템 elapsed
# 5.50   0.36   595.75

h2o.performance(dlearning.model)
# H2ORegressionMetrics:deeplearning
# ** Reported on training data. **
# Description:Metrics reported on temporary training frame with 10017 samples

# MSE:5782595
# R2:0.7611859
# Mean Residual Deviance:5782595

#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#create a data frame and writing submission file
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.dl2$predict)
write.csv(sub_dlearning, file = "C:/Users/bevis/Downloads/Black_Friday/sub_dlearning_new.csv", row.names = F)

####_5. Next To Do ------------------------
# GBM, Deep Learning 및 Random Forest에서 매개 변수 튜닝을 수행하십시오.
# 매개 변수 튜닝에 그리드 검색을 사용하십시오. H2O에는이 작업을 수행하는 h2o.grid라는 멋진 함수가 있습니다.
# 모델에 새로운 정보를 가져올 수있는 더 많은 기능을 만드는 방법을 생각해보십시오.
# 마지막으로, 더 좋은 모델을 얻기 위해 모든 결과를 앙상블 처리하십시오.

