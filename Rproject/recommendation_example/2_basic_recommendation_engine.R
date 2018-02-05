####--------------------------------------- 
# Building our basic recommendation engine 
####--------------------------------------- 

####--------------------------------------- 
## 1. Loading and formatting data

ratings = read.csv("https://raw.githubusercontent.com/sureshgorakala/RecommenderSystems_R/master/movie_rating.csv")

head(ratings)

dim(ratings)

str(ratings)

## ----------
# 추천 시스템을 구축하려면 행에 사용자가 포함된 행렬이 필요
# 추천 시스템 엔진을 구축에 맞춰 데이터를 변환 
# 영화 제목을 row, 평론가를 column으로 행렬 형식으로 변환
# 데이터 변환에는 acast 사용
## ----------

# data processing and formatting
if(!require("reshape2")) { install.packages("reshape2")}
movie_ratings = as.data.frame(acast(ratings, title~critic,
                                    value.var="rating"))

movie_ratings

####--------------------------------------- 
## 2. Calculating similarity between users

## ----------
# 사용자간 유사성 척도로 상관 관계를 사용
  # 상관 관계가 두 항목의 연관성,두 항목 벡터가 얼마나 공존했는지,
  # 서로 관련이 있는지를 나타내기 때문에 이용
## ----------

## cor 참고 사이트 - http://bwlewis.github.io/covar/missing.html
## complete.obs 는 NA가 있는 행을 삭제하고 상관계수 계산
sim_users_com = cor(movie_ratings[,1:6],use="complete.obs")
sim_users_com

## complete.obs 는 NA가 있는 행에 알려지지 않은 미지수를 변환하여 상관계수 계산 
sim_users_pair = cor(movie_ratings[,1:6],use="pairwise.complete.obs")
sim_users_pair

## NA 대체 추천
# NA값을 가진 열의 "평균값"
# 시계열 및 정렬된 데이터에서는 "마지막 값 이월" 
# "누락되지 않은 값에서 부트스트랩 추천"
# 많은 관찰 데이터가 있는 경우 - "기본 클러스터링 알고리즘으로 파티셔닝 후 클러스터 코호트에서 누락된 값 입력"

####--------------------------------------- 
## 3. Predicting the unknown ratings for users.

## step ----------
# Toby가 평가하지 않은 영화를 추출
# 이 영화들에 대해 다른 비평가들이 준 모든 등급을 분리
# 이 영화에 주어진 등급에 Toby 이외의 모든 비평가와 Toby와의 비평가의 유사 가치를 곱함
# 각 영화의 총 평점을 합산하고 이 합계 된 값의 유사성 평론 값 합으로 나눔

if(!require("data.table")) { install.packages("data.table")}
rating_critic = setDT(movie_ratings[colnames(movie_ratings)[6]],keep.rownames = TRUE)[]
names(rating_critic) = c('title','rating')
View(rating_critic)

# Toby 평점이 없는 영화 추출
titles_na_critic = rating_critic$title[is.na(rating_critic$rating)]
titles_na_critic

# Toby 평점이 없는 영화에 다른 비평가의 평점 추출 
ratings_t = ratings[ratings$title %in% titles_na_critic,]
ratings_t

# Toby와 다른 비평가의 유사성 계산
# add similarity values for each user as new variable
x = (setDT(data.frame(sim_users_com[,6]),keep.rownames = TRUE)[])
names(x) = c('critic','similarity')

# Toby 평점이 없는 영화에 다른 비평가의 평점 + Toby와 다른 비평가 유사성 열 추가 
ratings_t = merge(x = ratings_t, y = x, by = "critic", all.x = TRUE)
ratings_t

# sim_rating 열 추가 = rating * similarity
# mutiply rating with similarity values
ratings_t$sim_rating = ratings_t$rating*ratings_t$similarity
ratings_t

## 4. Recommending items to users based on user-similarity score.
# Toby 평점이 없는 영화 평점 예측
# predicting the non rated titles
if(!require("dplyr")) { install.packages("dplyr")}
result = ratings_t %>% 
         group_by(title) %>%
         summarise(sum(sim_rating)/sum(similarity))
result

# Toby 평점이 없는 영화 예측 평점 평균 
mean(rating_critic$rating,na.rm = T)

# function to make recommendations
generateRecommendations <- function(userId){
  rating_critic = setDT(movie_ratings[colnames(movie_ratings)
                                      [userId]],keep.rownames = TRUE)[]
  names(rating_critic) = c('title','rating')
  titles_na_critic =
    rating_critic$title[is.na(rating_critic$rating)]
  ratings_t =ratings[ratings$title %in% titles_na_critic,]
  #add similarity values for each user as new variable
  x = (setDT(data.frame(sim_users_com[,userId]),keep.rownames = TRUE)
       [])
  names(x) = c('critic','similarity')
  ratings_t = merge(x = ratings_t, y = x, by = "critic", all.x =
                      TRUE)
  #mutiply rating with similarity values
  ratings_t$sim_rating = ratings_t$rating*ratings_t$similarity
  #predicting the non rated titles
  result = ratings_t %>% group_by(title) %>%
    summarise(sum(sim_rating)/sum(similarity))
  return(result)
}

