

library(bitops)
# loading csv
library(RCurl)
library(tidyr)
# 데이터 구조
library(dplyr)
library(knitr)
# plotting
library(ggplot2)
# 테이블 형식
library(DT)
library(pander)
library(Matrix)
library(recommenderlab)

# MovieLense 데이터 
data(MovieLense, package = "recommenderlab") 
movielense <- MovieLense  
class(movielense)
movielense
moviemeta <- MovieLenseMeta  # Binary data 
# head(moviemeta)  
pander(head(moviemeta), caption = "Sample Movie Meta Data") 


movielenseorig <- movielense 

# (유저) 20개 이상 평가 | (영화) 50개 이상의 평가
movielense <- movielense[rowCounts(movielense) > 20, colCounts(movielense) > 50] 
minrowcnt <- min(rowCounts(movielense)) 
movielense

set.seed(101) 
# 시드함수 생성하고 데이터 수와 같은 논리적 객체 생성, TRUE는 훈련세트, FALSE는 테스트 세트
which_train <- sample(x = c(TRUE, FALSE), size = nrow(movielense), replace = TRUE, prob = c(0.8, 0.2)) 
recc_data_train <- movielense[which_train, ] 
recc_data_test <- movielense[!which_train, ]

# IBCF 상위 10개의 영화 추천 모델 생성
recc_model1 <- Recommender(data = recc_data_train, method = "IBCF", parameter = list(k = 25, method = "Cosine"))
recc_model1

# 설정
num_rec <- 10  # 추천수 설정
# predict 함수 사용으로 사용자마다 10개의 추천을 생성하고 예측 수행
recc_predicted1 <- predict(object = recc_model1, newdata = recc_data_test, n = num_rec) 
recc_predicted1

recdf <- data.frame(user = sort(rep(1:length(recc_predicted1@items), recc_predicted1@n)), 
                    rating = unlist(recc_predicted1@ratings), index = unlist(recc_predicted1@items))

#### Recommendations from IBCF model
recdf$title <- recc_predicted1@itemLabels[recdf$index]
recdf$year <- moviemeta$year[recdf$index]
recdf <- recdf %>% group_by(user) %>% top_n(10, recdf$rating)
datatable(recdf[recdf$user %in% (1:10), ])

#### Recommendations from IBCF medel with period context added
recdfnew <- recdf[with(recdf, order(recdf$user, -recdf$year, -round(recdf$rating))), c(1, 2, 5, 4)]
recdfnew <- recdfnew %>% group_by(user) %>% top_n(10, recdfnew$year)
datatable(recdfnew[recdfnew$user %in% (1:10), ])

#  UBCF 상위 10개 추천 모델 생성
recc_model2 <- Recommender(data = recc_data_train, method = "UBCF", parameter = list(k = 25, method = "Cosine"))
recc_model2

# setting
num_rec <- 10  # 추천수 설정

recc_predicted2 <- predict(object = recc_model2, newdata = recc_data_test, n = num_rec)
recc_predicted2

recdfub <- data.frame(user = sort(rep(1:length(recc_predicted2@items), recc_predicted2@n)), 
                      rating = unlist(recc_predicted2@ratings), index = unlist(recc_predicted2@items))

#### Recommendations from UBCF model
recdfub$title <- recc_predicted2@itemLabels[recdfub$index] 
recdfub$year <- moviemeta$year[recdfub$index] 
recdfub <- recdfub %>% group_by(user) %>% top_n(5, recdfub$rating) 
datatable(recdfub[recdfub$user %in% (1:10), ])

#### Recommendations from UBCF medel with period context added
recdfubnew <- recdfub[with(recdfub, order(recdfub$user, -recdfub$year, -round(recdfub$rating))), c(1, 2, 5, 4)]
recdfubnew <- recdfubnew %>% group_by(user) %>% top_n(5, recdfubnew$year)
datatable(recdfubnew[recdfubnew$user %in% (1:10), ])

#  UBCF 상위 10개 추천 모델 생성
recc_model2 <- Recommender(data = recc_data_train, method = "UBCF", parameter = list(k = 25, method = "Cosine"))
recc_model2

# setting
num_rec <- 10  # 추천수 설정

recc_predicted2 <- predict(object = recc_model2, newdata = recc_data_test, n = num_rec)
recc_predicted2

recdfub <- data.frame(user = sort(rep(1:length(recc_predicted2@items), recc_predicted2@n)), 
                      rating = unlist(recc_predicted2@ratings), index = unlist(recc_predicted2@items))


#### Recommendations from UBCF model
recdfub$title <- recc_predicted2@itemLabels[recdfub$index] 
recdfub$year <- moviemeta$year[recdfub$index] 
recdfub <- recdfub %>% group_by(user) %>% top_n(5, recdfub$rating) 
datatable(recdfub[recdfub$user %in% (1:10), ])


#### Recommendations from UBCF medel with period context added
recdfubnew <- recdfub[with(recdfub, order(recdfub$user, -recdfub$year, -round(recdfub$rating))), c(1, 2, 5, 4)]
recdfubnew <- recdfubnew %>% group_by(user) %>% top_n(5, recdfubnew$year)
datatable(recdfubnew[recdfubnew$user %in% (1:10), ])

set.seed(101)
n_fold <- 10  # 교차검증 폴드 수
items_to_keep <- 15  # Items to consider in training set (less than min no of ratings )
rating_threshold <- 3.5  # Considering a rating of 3.5 as good rating across all movies

eval_sets <- evaluationScheme(data = movielense, method = "cross-validation", k = n_fold, 
                              given = items_to_keep, goodRating = rating_threshold)
eval_sets

evaltrain <- getData(eval_sets, "train")  # training set
evalknown <- getData(eval_sets, "known")  # known test set
evalunknown <- getData(eval_sets, "unknown")  # unknown test set

# IBCF

model_to_evaluate <- "IBCF"
model_parameters <- list(method = "Cosine")
model1_IBCF_cosine <- Recommender(data = evaltrain, method = model_to_evaluate, parameter = model_parameters)
items_to_recommend <- 10
model1_prediction <- predict(object = model1_IBCF_cosine, newdata = evalknown, n = items_to_recommend, 
                             type = "ratings")
model1_predtop <- predict(object = model1_IBCF_cosine, newdata = evalknown, n = items_to_recommend, 
                          type = "topNList")
model1_accuracy <- calcPredictionAccuracy(x = model1_prediction, data = evalunknown, 
                                          byUser = FALSE) # byUser =FALSE for model level performance metrics
model1_accuracy

# UBCF

model_to_evaluate <- "UBCF"
model_parameters <- list(method = "cosine")
model3_UBCF_cosine <- Recommender(data = evaltrain, method = model_to_evaluate, parameter = model_parameters)
items_to_recommend <- 10
model3_prediction <- predict(object = model3_UBCF_cosine, newdata = evalknown, n = items_to_recommend, 
                             type = "ratings")
model3_predtop <- predict(object = model3_UBCF_cosine, newdata = evalknown, n = items_to_recommend, 
                          type = "topNList")
model3_accuracy <- calcPredictionAccuracy(x = model3_prediction, data = evalunknown, 
                                          byUser = FALSE)  # byUser =FALSE 전체 모델의 정확도 계산
model3_accuracy

# 2개의 다른 모델을 평가하기 위해서 그래프 ROC 커브 구현
models_to_evaluate <- list(IBCF_cos = list(name = "IBCF", param = list(method = "cosine")), 
                           UBCF_cos = list(name = "UBCF", param = list(method = "cosine")))
# In order to evaluate the models properly, we need to test them, varying the
# number of flavors , as follows
n_recommendations <- c(1, 3, 5, 7, 10, 12, 15)
list_results <- evaluate(x = eval_sets, method = models_to_evaluate, n = n_recommendations)
plot(list_results, annotate = 1, legend = "topleft")
title("ROC curve")

plot(list_results, "prec/rec", annotate = 1, legend = "bottomright")
title("Precision-recall")





