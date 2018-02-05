####--------------------------------------- 
# Building Personalized Recommendation Engines
####--------------------------------------- 

####--------------------------------------- 
## 1. Personalized recommender systems
  ## Content-based recommendation using R
# Building a content-based recommendation system
  # 사용자 프로파일 생성
  # 품목 프로파일을 생성
  # 추천 엔진 모델을 생성
  # top N 추천 제안
raw_data = read.csv("C:/Users/bevis/Downloads/RecommendationEngines/udata.csv",sep=",",header=F)
colnames(raw_data) = c("UserId","MovieId","Rating","TimeStamp")
raw_data[1,1] <- 196

ratings = raw_data[,1:3]
head(ratings)

movies = read.csv("C:/Users/bevis/Downloads/RecommendationEngines/uitem.csv",sep=",",header=F)
colnames(movies) =  c("MovieId","MovieTitle","ReleaseDate","VideoReleaseDate","IMDbURL","Unknown",
                      "Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama",
                      "Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western")
movies[1,1] <- 1
movies$IMDbURL <- gsub('%20',' ', movies$IMDbURL)

movies = movies[,-c(2:5)]
View(movies)

ratings = merge(x = ratings, y = movies, by = "MovieId", all.x = TRUE)
View(ratings)

# Rating 1~3 = 0 / Rating 4~5 = 1 등급 레이블 추가 
nrat = unlist(lapply(ratings$Rating,function(x)
{
  if(x>3) {return(1)}
  else {return(0)}
}))

ratings = cbind(ratings,nrat)

apply(ratings[,-c(1:3,23)],2,function(x)table(x))  # MovieId,UserId,Rating,nrat

# 새로운 변수 nrat 생성하였으므로, rating 변수 제거
scaled_ratings = ratings[,-c(3,4)]

# data scale. 표준화 또는 가운데 맞춤. scale() 함수 이용
scaled_ratings=scale(scaled_ratings[,-c(1,2,21)])
scaled_ratings = cbind(scaled_ratings,ratings[,c(1,2,23)])

# train - data 80% / test - data 20% 분할 
set.seed(7)
which_train <- sample(x = c(TRUE, FALSE), size = nrow(scaled_ratings),
                      replace = TRUE, prob = c(0.8, 0.2))
model_data_train <- scaled_ratings[which_train, ]
model_data_test <- scaled_ratings[!which_train, ]

if(!require("randomForest")) { install.packages("randomForest")}
fit = randomForest(as.factor(nrat)~., data = model_data_train[,-c(19,20)]) # missing value로 에러 
summary(fit)

predictions <- predict(fit, model_data_test[,-c(19,20,21)], type="class")

#building confusion matrix
cm = table(predictions,model_data_test$nrat)
(accuracy <- sum(diag(cm)) / sum(cm))
(precision <- diag(cm) / rowSums(cm))
recall <- (diag(cm) / colSums(cm))

# 1. Create a DataFrame containing all the movies not rated by the active user (user id: 943 in our case).
#extract distinct movieids
totalMovieIds = unique(movies$MovieId)

#see the sample movieids using tail() and head() functions:
#a function to generate dataframe which creates non-rated movies by active user and set rating to 0;
nonratedmoviedf = function(userid){ratedmovies = raw_data[raw_data$UserId==userid,]$MovieId
non_ratedmovies = totalMovieIds[!totalMovieIds %in%
                                  ratedmovies]
df = data.frame(cbind(rep(userid),non_ratedmovies,0))
names(df) = c("UserId","MovieId","Rating")
return(df)
}

# let's extract non-rated movies for active userid 943
activeusernonratedmoviedf = nonratedmoviedf(943)

# 2. Build a profile for this active user DataFrame:
activeuserratings = merge(x = activeusernonratedmoviedf, y = movies, by = "MovieId", all.x = TRUE)

# 3. Predict ratings, sort and generate 10 recommendations:
#use predict() method to generate predictions for movie ratings by the active user profile created in the previous step.
predictions <- predict(fit, activeuserratings[,-c(1:4)], type="class")

#creating a dataframe from the results
recommend = data.frame(movieId = activeuserratings$MovieId,predictions)

#remove all the movies which the model has predicted as 0 and then we can use the remaining items 
# as more probable movies which might be liked by the active user.
recommend = recommend[which(recommend$predictions == 1),]

####--------------------------------------- 
## 1. Personalized recommender systems
  ## Context-aware recommendations using R
# Context-aware recommender systems
  # context 정의
  # 사용자 item content에 대한 context profile 생성
  # context에 대한 권장 사항 생성

# Defining for context
raw_data = read.csv("C:/Users/bevis/Downloads/RecommendationEngines/udata.csv",sep=",",header=F)
colnames(raw_data) = c("UserId","MovieId","Rating","TimeStamp")
raw_data[1,1] <- 196

movies = read.csv("C:/Users/bevis/Downloads/RecommendationEngines/uitem.csv",sep=",",header=F)
colnames(movies) =  c("MovieId","MovieTitle","ReleaseDate","VideoReleaseDate","IMDbURL","Unknown",
                      "Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama",
                      "Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi","Thriller","War","Western")
movies[1,1] <- 1
movies$IMDbURL <- gsub('%20',' ', movies$IMDbURL)

movies = movies[,-c(2:5)]

ratings_ctx = merge(x = raw_data, y = movies, by = "MovieId", all.x = TRUE)

# Creating context profile
ts = ratings_ctx$TimeStamp

hours <- as.POSIXlt(ts,origin="1960-10-01")$hour
ratings_ctx = data.frame(cbind(ratings_ctx,hours))

 # 사용자 Id : 943의 context profile 작성
UCP = ratings_ctx[(ratings_ctx$UserId == 943),][,-c(2,3,4,5)]

 # 사용자 Id : 943의 매시간 동안 모든 항목 기능 계산
UCP_pref = aggregate(.~hours,UCP[,-1],sum)

 # 사용자 Id : 943의 9시간동안 시청한 영화 정규화
UCP_pref_sc = cbind(context = UCP_pref[,1],t(apply(UCP_pref[,-1], 1,
                                                   function(x)(x-min(x))/(max(x)-min(x)))))

# Generating context-aware recommendations
recommend$MovieId

 # recommendations와 movies dataset merge
UCP_pref_content = merge(x = recommend, y = movies, by = "MovieId", all.x = TRUE)

 # 사용자 콘텐츠 권장 사항 및 사용자의 9 시간 컨텍스트 기본 설정에 대한 요소별 승수 생성 :
active_user =cbind(UCP_pref_content$MovieId,(as.matrix(UCP_pref_content[,-c(1,2,3)]) %*% as.matrix(UCP_pref_sc[4,2:19])))
# UCP_pref_content : - MovieId, predictions, unknown
# UCP_pref_sc : Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, FilmNoir, Horror, Musical,
#               Mystery, Romance, SciFi, Thriller, War,  Western
head(active_user)

 # We can create a dataframe object of the prediction object:
active_user_df = as.data.frame(active_user)

 # Next, we add column names to the predictions object:
names(active_user_df) = c('MovieId','SimVal')

 # Then we sort the results:
FinalPredicitons_943 = active_user_df[order(-active_user_df$SimVal),]