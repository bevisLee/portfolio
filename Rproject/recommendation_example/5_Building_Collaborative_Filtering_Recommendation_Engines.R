####--------------------------------------- 
# Building Collaborative Filtering Recommendation Engines
####--------------------------------------- 

####--------------------------------------- 
## 1. The Jester5k dataset we will be using for this chapter
if(!require("recommenderlab")) { install.packages("recommenderlab")}

data_package <- data(package = "recommenderlab")
data_package$results[,c("Item","Title")]

data(Jester5k)

class(Jester5k)
object.size(Jester5k)
#convert the real-rating matrix into R matrix
object.size(as(Jester5k,"matrix"))

object.size(as(Jester5k, "matrix"))/object.size(Jester5k)

methods(class = class(Jester5k))

####--------------------------------------- 
## 2. Exploring the dataset and understanding the data
dim(Jester5k)
class(Jester5k@data)

# Exploring the rating values
hist(getRatings(Jester5k), main="Distribution of ratings")

####--------------------------------------- 
## 3. Building user-based collaborative filtering in R
if(!require("recommenderlab")) { install.packages("recommenderlab")}
data(Jester5k)

head(as(Jester5k,"matrix")[,1:10])
  # data의 80% - train data로 활용 / data의 20% - test data로 활용
  # k-fold cross-validation approach model을 추천 모델을 평가
  # 추천 모델 Parameter tuning

# Preparing training and test data
set.seed(1)
which_train <- sample(x = c(TRUE, FALSE), size = nrow(Jester5k),replace = TRUE,prob = c(0.8, 0.2))
head(which_train)

rec_data_train <- Jester5k[which_train, ]
dim(rec_data_train)

rec_data_test <- Jester5k[!which_train, ]
dim(rec_data_test)

# Creating a user-based collaborative model
recommender_models <- recommenderRegistry$get_entries(dataType = "realRatingMatrix")
recommender_models

recc_model <- Recommender(data = rec_data_train, method = "UBCF")
recc_model

recc_model@model$data

# Predictions on the test set
n_recommended <- 10
recc_predicted <- predict(object = recc_model,newdata = rec_data_test, n = n_recommended)
recc_predicted

# Let's define list of predicted recommendations:
rec_list <- sapply(recc_predicted@items, function(x){
  colnames(Jester5k)[x]
})

class(rec_list)

rec_list [1:2]

number_of_items = sort(unlist(lapply(rec_list, length)),decreasing = TRUE)
table(number_of_items)

# Analyzing the dataset
table(rowCounts(Jester5k))

model_data = Jester5k[rowCounts(Jester5k) < 80]
dim(model_data)

boxplot(model_data) # 에러 : NA 포함 
boxplot(rowMeans(model_data [rowMeans(model_data)>=-5 & rowMeans(model_data)<=7]))

model_data = model_data [rowMeans(model_data)>=-5 & rowMeans(model_data)<= 7]
dim(model_data)
image(model_data, main = "Rating distribution of 100 users") # 에러 : 그래프 제대로 출력 안됨

# Evaluating the recommendation model using the k-cross validation
items_to_keep <- 30
rating_threshold <- 3
n_fold <- 5 # 5-fold
eval_sets <- evaluationScheme(data = model_data, method = "cross-validation",
                              train = percentage_training, given = items_to_keep, goodRating = rating_threshold, k = n_fold)

size_sets <- sapply(eval_sets@runsTrain, length)
size_sets

  # train: This is the training set
  # known: This is the test set, with the item used to build the recommendations
  # unknown: This is the test set, with the item used to test the recommendations
getData(eval_sets, "train")
getData(eval_sets, "known")
getData(eval_sets, "unknown")

# Evaluating user-based collaborative filtering
model_to_evaluate <- "UBCF"
model_parameters <- NULL

eval_recommender <- Recommender(data = getData(eval_sets, "train"),
                                method = model_to_evaluate, parameter = model_parameters)

items_to_recommend <- 10
eval_prediction <- predict(object = eval_recommender, 
                           newdata = getData(eval_sets, "known"), n = items_to_recommend, type = "ratings")
eval_prediction

eval_accuracy <- calcPredictionAccuracy(x = eval_prediction, 
                                        data = getData(eval_sets, "unknown"), byUser = TRUE)
head(eval_accuracy)

apply(eval_accuracy,2,mean)

eval_accuracy <- calcPredictionAccuracy(x = eval_prediction, 
                                        data = getData(eval_sets, "unknown"), byUser = FALSE)
eval_accuracy

results <- evaluate(x = eval_sets, method = model_to_evaluate, n = seq(10, 100,10))
head(getConfusionMatrix(results)[[1]])
  # TP(True Positivie) : 정확하게 평가된 추천 품목 / True 인데, True라고 맞춘 경우(잘한 경우)
  # FP(False Positive) : 추천하지 않은 항목 / False 인데, True라고 한 경우(틀렸어요)
  # TN(True Negative)  : 등급이 매겨지지 않은 항목으로 추천 상품이 아니라고 예측 / False 인데, False라고 맞춘 경우(잘한 경우)
  # FN(False Negative) : 추천 항목으로 추천 상품이 아니라고 예측 / True 인데 False 라고 한 경우(틀렸어요)
  # 완벽한(또는 overfitted) 모델에는 TP와 TN 갖음 

  # Accuracy	TP+TN/TP+TN+FP+FN	
  # Precision	TP/TP+FP	
  # Recall	TP/TP+FN

columns_to_sum <- c("TP", "FP", "FN", "TN")
indices_summed <- Reduce("+", getConfusionMatrix(results))[, columns_to_sum]
head(indices_summed)

plot(results, annotate = TRUE, main = "ROC curve")
  # TPR(True Positive Rate) / FPR(False Positive Rate)  
  # TPR과 FPR 간의 균형을 유지하는 방식으로 값을 선택
  # TPR은 0.7에 가깝고, FPT은 0.4이고, nn=40으로 이동할때 
  # TPR은 여전히 0.7에 가깝지만, FPR은 0.7에 가깝기 때문에 nn=30은 좋은 트레이드 오프


####--------------------------------------- 
## 4. Building item-based collaborative filtering in R
if(!require("recommenderlab")) { install.packages("recommenderlab")}
data(Jester5k)

model_data = Jester5k[rowCounts(Jester5k) < 80]
model_data

boxplot(rowMeans(model_data))

dim(model_data[rowMeans(model_data) < -5])

dim(model_data[rowMeans(model_data) > 7])

model_data = model_data [rowMeans(model_data)>=-5 & rowMeans(model_data)<= 7]
model_data
  # train data & test data를 사용하여 IBCF recommender 모델 작업
  # 모델 평강 (Evaluating the model)
  # Parameter tuning

# Building an IBCF recommender model
which_train <- sample(x = c(TRUE, FALSE), size = nrow(model_data),
                      replace = TRUE, prob = c(0.8, 0.2))
class(which_train)

head(which_train)

model_data_train <- model_data[which_train, ]
dim(model_data_train)

model_data_test <- model_data[!which_train, ]
dim(model_data_test)

model_to_evaluate <- "IBCF"
model_parameters <- list(k = 30)

model_recommender <- Recommender(data = model_data_train,method =model_to_evaluate, parameter = model_parameters)
model_recommender

items_to_recommend <- 10
model_prediction <- predict(object = model_recommender, 
                            newdata = model_data_test, n = items_to_recommend)
model_prediction
print(class(model_prediction))

slotNames(model_prediction)

model_prediction@items[[1]]

recc_user_1 = model_prediction@items[[1]]
jokes_user_1 <- model_prediction@itemLabels[recc_user_1]
jokes_user_1

# Model evaluation
n_fold <- 4
items_to_keep <- 15
rating_threshold <- 3

eval_sets <- evaluationScheme(data = model_data, method = "cross-validation",
                              k = n_fold, given = items_to_keep, goodRating =rating_threshold)
size_sets <- sapply(eval_sets@runsTrain, length)
size_sets

model_to_evaluate <- "IBCF"
model_parameters <- NULL

getData(eval_sets,"train")

eval_recommender <- Recommender(data = getData(eval_sets, "train"),
                                method = model_to_evaluate, parameter = model_parameters)

#setting the number of items to be set for recommendations
items_to_recommend <- 10

eval_prediction <- predict(object = eval_recommender, 
                           newdata = getData(eval_sets, "known"), n = items_to_recommend, type = "ratings")
class(eval_prediction)

# Model accuracy using metrics
eval_accuracy <- calcPredictionAccuracy(x = eval_prediction, 
                                        data = getData(eval_sets, "unknown"), byUser = TRUE)
head(eval_accuracy)

apply(eval_accuracy,2,mean)

eval_accuracy

# Model accuracy using plots
results <- evaluate(x = eval_sets, method = model_to_evaluate, 
                    n = seq(10,100,10))

results@results[1]

columns_to_sum <- c("TP", "FP", "FN", "TN","precision","recall")
indices_summed <- Reduce("+", getConfusionMatrix(results))[, columns_to_sum]

plot(results, annotate = TRUE, main = "ROC curve")
plot(results, "prec/rec", annotate = TRUE, main = "Precision-recall")

# Parameter tuning for IBCF
vector_k <- c(5, 10, 20, 30, 40)

model1 <- lapply(vector_k, function(k,l){ list(name = "IBCF", 
                                               param = list(method = "cosine", k = k)) })
names(model1) <- paste0("IBCF_cos_k_", vector_k)
names(model1) [1] 

#use Pearson method for similarities 
model2 <- lapply(vector_k, function(k,l){ list(name = "IBCF", 
                                     param = list(method = "pearson", k = k)) })

names(model2) <- paste0("IBCF_pea_k_", vector_k)
names(model2) [1]

#now let's combine all the methods:
models = append(model1,model2)

n_recommendations <- c(1, 5, seq(10, 100, 10))

list_results <- evaluate(x = eval_sets, method = models, n= n_recommendations)

plot(list_results, annotate = c(1,2), legend = "topleft")
title("ROC curve")

plot(list_results, "prec/rec", annotate = 1, legend = "bottomright")
title("Precision-recall")
