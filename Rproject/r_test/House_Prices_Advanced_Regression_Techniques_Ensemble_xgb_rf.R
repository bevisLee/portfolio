## 에러로 다시 확인 필요

x.train <- read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train_eda.csv")

x.test <- read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test_eda.csv")

x.train <- x.train[,-1]
x.test <- x.test[, -1]

if (!require("caret")) { install.packages("caret") }

# We split the data set into 2 parts: training data (90%) and testing data (20%).
inTrain <- createDataPartition(y = x.train$SalePrice,
                               p = 0.90,
                               list = FALSE)

num.variables <- dim(x.train)[2]

train.xgboost <- x.train[inTrain,]

test.xgboost <- x.train[-inTrain,]

### xgboost test
searchGridSubCol <- expand.grid(subsample = c(0.5, 0.75, 1),
                                colsample_bytree = c(0.6, 0.8, 1))
ntrees <- 100

#Build a xgb.DMatrix object
DMMatrixTrain <- xgb.DMatrix(data = data.matrix(train.xgboost[, - num.variables]),
                         label = data.matrix(train.xgboost[, num.variables]))

rmseErrorsHyperparameters <- apply(searchGridSubCol, 1, function(parameterList) {

    #Extract Parameters to test
    currentSubsampleRate <- parameterList[["subsample"]]
    currentColsampleRate <- parameterList[["colsample_bytree"]]

    xgboostModelCV <- xgb.cv(data = DMMatrixTrain, nrounds = ntrees, nfold = 5, showsd = TRUE,
                           metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
                           "objective" = "reg:linear", "max.depth" = 15, "eta" = 2 / ntrees,
                           "subsample" = currentSubsampleRate, "colsample_bytree" = currentColsampleRate)

    xvalidationScores <- xgboostModelCV #as.data.frame(xgboostModelCV)
    #Save rmse of the last iteration
    rmse <- tail(xvalidationScores$test.rmse.mean, 1)

    return(c(rmse, currentSubsampleRate,currentColsampleRate))

})

if (!require("xgboost")) { install.packages("xgboost") }
house.xgboost <- xgboost(data = data.matrix(train.xgboost[, - num.variables]),
                         label = data.matrix(train.xgboost[, num.variables]),
                 booster = "gblinear",
                 objective = "reg:linear",
                 max.depth = 20,
                 nround = 10000,
                 lambda = 0,
                 lambda_bias = 0,
                 alpha = 0,
                 missing = NA,
                 verbose = 0)

pred.test.xgboost <- predict(house.xgboost, data.matrix(test.xgboost), missing = NA)

rmse <- sqrt(sum((log(pred.test.xgboost) - log(test.xgboost$SalePrice)) ^ 2, na.rm = TRUE) / length(pred.test.xgboost))


## XGBoost
xgbModel <- function(dfTrain, dfTest) {
    test_names <- names(dfTest)
    train_names <- names(dfTrain)
    intersect_names <- intersect(test_names, train_names)

    training <- xgb.DMatrix(data = data.matrix(dfTrain[intersect_names]), label = (dfTrain[, c("SeriousDlqin2yrs")]))

    # subsample - setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow 
    # trees and this will prevent overfitting. 
    # colsample_bytree - subsample ratio of columns when constructing each tree.
    param <- list(objective = "binary:logistic", booster = "gbtree", eta = 0.01, max_depth = 50,
                subsample = 0.5, colsample_bytree = 0.5)

    train.xgb <- xgboost(data = training, params = param, nrounds = 500, verbose = 1, eval_metric = "auc")

    train.xgb
}

## GBM  : 에러 확인 필요
require(caret)

caretGrid <- expand.grid(interaction.depth = c(5, 10), n.trees = 500,
                   shrinkage = c(0.01, 0.001), n.minobsinnode = 5)

metric <- "ROC"

trainControl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

gbm.caret <- train(SalePrice ~ ., data = x.train, distribution = "bernoulli", method = "gbm",
              trControl = trainControl, verbose = TRUE,
              tuneGrid = caretGrid, metric = metric, bag.fraction = 0.3)

# Stochastic Gradient Boosting 
# 
# 150000 samples
#     10 predictor
#      2 classes: 'Good', 'Bad' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 135000, 135000, 134999, 134999, 135001, 135000, ... 
# Resampling results across tuning parameters:
# 
#   shrinkage  interaction.depth  Accuracy   Kappa    
#   0.001       5                 0.9331600  0.0000000
#   0.001      10                 0.9331600  0.0000000
#   0.010       5                 0.9374467  0.2608611
#   0.010      10                 0.9376333  0.2711224
# 
# Tuning parameter 'n.trees' was held constant at a value of 500
# Tuning parameter 'n.minobsinnode' was
#  held constant at a value of 5
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 500, interaction.depth = 10, shrinkage = 0.01
#  and n.minobsinnode = 5.

library(dplyr)
library(e1071)
library(randomForest)
library(gbm)
library(xgboost)

Train <- a.train
Test <- x_test
Train$X <- NULL
Test$X <- NULL
Test$SalePrice <- NULL

## Gradient Boosting Machine model as defined above in model tuning section
#    args: dfTrain -> the training dataframe required
# --------------------------------------------------------------
GBMModel <- function(dfTrain) {
    GB <- gbm(dfTrain$SalePrice ~ ., data = dfTrain, n.trees = 5000,
            keep.data = FALSE, shrinkage = 0.01, bag.fraction = 0.3,
            interaction.depth = 10)
    GB
}

## Random Forest model
#    args: dfTrain -> the training dataframe required
# --------------------------------------------------------------
forestModel <- function(dfTrain) {
    train.Forest <- randomForest(formula = SalePrice ~ .,
                               data = dfTrain, ntree = 2000, importance = TRUE,
                               na.action = na.omit)
    train.Forest
}

## Support Vector Machine model
#    args: dfTrain -> the training dataframe required
# --------------------------------------------------------------
svmModel <- function(dfTrain) {
    train.svm <- svm(SalePrice ~ .,
                   type = 'C-classification', kernel = 'radial',
                   cachesize = 2000, probability = TRUE, cost = 1,
                   data = dfTrain)
    train.svm
}

## Support Vector Machine model
#    args: dfTrain -> the training dataframe required
#          dfTest -> the test dataframe required
# --------------------------------------------------------------
xgbModel <- function(dfTrain, dfTest) {
    test_names <- names(dfTest)
    train_names <- names(dfTrain)
    intersect_names <- intersect(test_names, train_names)

    training <- xgb.DMatrix(data = data.matrix(dfTrain[intersect_names]), label = (dfTrain[, c("SalePrice")]))

    # subsample - setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow 
    # trees and this will prevent overfitting.
    # colsample_bytree - subsample ratio of columns when constructing each tree.
    param <- list(objective = "reg:linear", booster = "gbtree", eta = 0.01, max_depth = 50,
                subsample = 0.5, colsample_bytree = 0.5)

    train.xgb <- xgboost(data = training, params = param, nrounds = 500, verbose = 1, eval_metric = "auc")

    train.xgb
}

## Ensemble model
#    args: dfTrain -> the training dataframe required
#          dfTest -> the test dataframe required
# --------------------------------------------------------------
ensemble <- function(dfTrain, dfTest) {
    #SVM commented out because does not improve AUC score
    models <- list(rfs = forestModel(dfTrain),
                           gbs = GBMModel(dfTrain),
    #svm=svmModel(dfTrain),
                           xgb = xgbModel(dfTrain, dfTest))
    models
}

## Support Vector Machine model
#    args: model -> the ensemble model used to predict
#          dfTrain -> the training dataframe required
#          dfTest -> the test dataframe required
# --------------------------------------------------------------
ensemblePredictions <- function(model, train, test) {
    pred <- predict(model$rfs, test, type = 'prob')[, 2]
    pred2 <- predict(model$gbs, test, n.trees = 5000)
    #pred3 <- predict(model$svm, test)

    test_names <- names(test)
    train_names <- names(train)
    intersect_names <- intersect(test_names, train_names)

    testDataXGB <- xgb.DMatrix(data = data.matrix(test[, intersect_names]))
    pred4 <- predict(xgb1, testDataXGB)

    pred <- data.frame(Id = test_id, ProbabilityRF = pred,
                     ProbabilityGBM = pred2, ProbabilityXGB = pred4)
    pred$ProbabilityGBM <- 1 / (1 + exp(-pred$ProbabilityGBM)) # sigmoid function to convert to probability

    # After submitting each model individually, this upweights models with greater individual AUC
    pred$Probability <- ((pred$ProbabilityGBM) * 4 + pred$ProbabilityRF + (pred$ProbabilityXGB) * 20) / 25

    pred <- pred %>% dplyr::select(Id, Probability)
    write.csv(pred,paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/Ensemble_model_", format(Sys.time(), "%Y%b%d_%H%M%S")), row.names = FALSE)
}

ensemble.models <- ensemble(Train, Test)
ensemblePredictions(ensemble.models, Train, Test)

