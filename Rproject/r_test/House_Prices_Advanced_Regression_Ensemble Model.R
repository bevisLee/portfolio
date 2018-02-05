####_Step.0_data import --------------------------------------------------
참조 - https://www.kaggle.com/jimthompson/ensemble-model-stacked-model-example

if (!require("plyr")) { install.packages("plyr") }
if (!require("dplyr")) { install.packages("dplyr") }
if (!require("xgboost")) { install.packages("xgboost") }
if (!require("ranger")) { install.packages("ranger") }
if (!require("nnet")) { install.packages("nnet") }
if (!require("Metrics")) { install.packages("Metrics") }
if (!require("ggplot2")) { install.packages("ggplot2") }

train <- read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train.csv", stringsAsFactors = T)
test <- read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test.csv", stringsAsFactors = T)

####_Step.1_incorporate results of Boruta analysis ------------------------
CONFIRMED_ATTR <- c("MSSubClass", "MSZoning", "LotArea", "LotShape", "LandContour", "Neighborhood",
                    "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt",
                    "YearRemodAdd", "Exterior1st", "Exterior2nd", "MasVnrArea", "ExterQual",
                    "Foundation", "BsmtQual", "BsmtCond", "BsmtFinType1", "BsmtFinSF1",
                    "BsmtFinType2", "BsmtUnfSF", "TotalBsmtSF", "HeatingQC", "CentralAir",
                    "X1stFlrSF", "X2ndFlrSF", "GrLivArea", "BsmtFullBath", "FullBath", "HalfBath",
                    "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional",
                    "Fireplaces", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish",
                    "GarageCars", "GarageArea", "GarageQual", "GarageCond", "PavedDrive", "WoodDeckSF",
                    "OpenPorchSF", "Fence") # 50개 확인된 변수

TENTATIVE_ATTR <- c("Alley", "LandSlope", "Condition1", "RoofStyle", "MasVnrType", "BsmtExposure",
                    "Electrical", "EnclosedPorch", "SaleCondition") # 9개 시험 변수

REJECTED_ATTR <- c("LotFrontage", "Street", "Utilities", "LotConfig", "Condition2", "RoofMatl",
                   "ExterCond", "BsmtFinSF2", "Heating", "LowQualFinSF", "BsmtHalfBath",
                   "X3SsnPorch", "ScreenPorch", "PoolArea", "PoolQC", "MiscFeature", "MiscVal",
                   "MoSold", "YrSold", "SaleType") # 20개 거절된 변수

PREDICTOR_ATTR <- c(CONFIRMED_ATTR, TENTATIVE_ATTR, REJECTED_ATTR) # Id 제외한 전체 79개 변수

# Determine data types in the data set
data_types <- sapply(PREDICTOR_ATTR, function(x) { class(train[[x]]) })
unique_data_types <- unique(data_types)

# Separate attributes by data type
DATA_ATTR_TYPES <- lapply(unique_data_types, function(x) { names(data_types[data_types == x]) }) # 79개 변수 colname & data type 저장
names(DATA_ATTR_TYPES) <- unique_data_types # 79개 data type 중복제거 저장 : "integer" / "factor" 

# create folds for training
if (!require("caret")) { install.packages("caret") }

set.seed(13)
data_folds <- createFolds(train$SalePrice, k = 5) # train data -> random Fold 5개 list로 저장

####_Step.2-1_Feature Set 1(L0FeatureSet1) - Boruta Confirmed and tentative Attributes ------------------------
prepL0FeatureSet1 <- function(df) {
    id <- df$Id
    if (class(df$SalePrice) != "NULL") {
        y <- log(df$SalePrice)
    } else {
        y <- NULL
    } # SalePrice 값이 있으면 log 치환, 없으면 Null 입력
    
    predictor_vars <- c(CONFIRMED_ATTR, TENTATIVE_ATTR) # 59개 변수(확인+시험)

    # predictors <- data.table(df,predictor_vars) # data.table에 적용
    predictors <- df[predictor_vars] # 입력받은 df에 59개 변수 data 저장

    # for numeric set missing values to -1 for purposes
    num_attr <- intersect(predictor_vars, DATA_ATTR_TYPES$integer) # 59개 변수 중 data type이 interger인 변수만 num_attr 저장 : 26개
    # "MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd" 
    # "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF", "X1stFlrSF", "X2ndFlrSF"
    # "GrLivArea", "BsmtFullBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr"
    # "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF"
    # "OpenPorchSF", "EnclosedPorch"
    for (x in num_attr) {
        predictors[[x]][is.na(predictors[[x]])] <- -1
    } # 26개 변수 중 data값이 NA일 경우 -1 입력

    # for character  atributes set missing value
    char_attr <- intersect(predictor_vars, DATA_ATTR_TYPES$factor) # 59개 변수 중 data type이 character인 변수만 추출하여 저장 : character형이 없어 NULL / character이 없어 factor로 변경

    for (x in char_attr) {
        predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
        predictors[[x]] <- factor(predictors[[x]])
    } # 26개 변수 중 data값이 NA일 경우 "*MISSING*" 입력

    return(list(id = id, y = y, predictors = predictors))
}

L0FeatureSet1 <- list(train = prepL0FeatureSet1(train),
                    test = prepL0FeatureSet1(test))

####_Step.2-2_Feature Set 2(xgboost/prepL0FeatureSet2) - Boruta Confirmed Attributes ------------------------
prepL0FeatureSet2 <- function(df) {
    id <- df$Id
    if (class(df$SalePrice) != "NULL") {
        y <- log(df$SalePrice)
    } else {
        y <- NULL
    }    # SalePrice 값이 있으면 log 치환, 없으면 Null 입력
    
    predictor_vars <- c(CONFIRMED_ATTR, TENTATIVE_ATTR) # 59개 변수(확인+시험)

#    predictors <- data.table(df, predictor_vars) # data.table에 적용
    predictors <- df[predictor_vars] # 입력받은 df에 59개 변수 data 저장

    # for numeric set missing values to -1 for purposes
    num_attr <- intersect(predictor_vars, DATA_ATTR_TYPES$integer)
    for (x in num_attr) {
        predictors[[x]][is.na(predictors[[x]])] <- -1
    } # 26개 변수 중 data값이 NA일 경우 -1 입력

    # for character  atributes set missing value
    char_attr <- intersect(predictor_vars, DATA_ATTR_TYPES$character) # 59개 변수 중 data type이 character인 변수만 추출하여 저장 : character형이 없어 NULL
    for (x in char_attr) {
        predictors[[x]][is.na(predictors[[x]])] <- "*MISSING*"
        predictors[[x]] <- as.numeric(factor(predictors[[x]]))
    } # 26개 변수 중 data값이 NA일 경우 "*MISSING*" 입력

    return(list(id = id, y = y, predictors = as.matrix(predictors)))
}

L0FeatureSet2 <- list(train = prepL0FeatureSet2(train),
                    test = prepL0FeatureSet2(test))

####_Step.3-1_trainOneFold_train model on one data fold ------------------------
trainOneFold <- function(this_fold, feature_set) {
    # get fold specific cv data
    cv.data <- list() # cv.data list() 생성
    cv.data$predictors <- feature_set$train$predictors[this_fold,] # cv.data$predictors 에 train$predictors 중 data_folds에 해당하는 row 값 입력
    cv.data$ID <- feature_set$train$id[this_fold] # cv.data$ID 에 train$id 중 data_folds에 해당하는 row 값 입력
    cv.data$y <- feature_set$train$y[this_fold] # cv.data$y 에 train$y(SalePrice) 중 data_folds에 해당하는 row 값 입력

    # get training data for specific fold
    train.data <- list() # train.data list() 생성
    train.data$predictors <- feature_set$train$predictors[-this_fold,] # train.data$predictors 에 train$predictors 중 data_folds에 해당하는 row 제외 값 입력
    train.data$y <- feature_set$train$y[-this_fold] # train.data$y 에 train$y(SalePrice) 중 data_folds에 해당하는 row 제외 값 입력

    set.seed(825) # 난수 고정(825)
    fitted_mdl <- do.call(train,
                          c(list(x = train.data$predictors, y = train.data$y),
                        CARET.TRAIN.PARMS,
                        MODEL.SPECIFIC.PARMS,
                        CARET.TRAIN.OTHER.PARMS)) # do.call 명령문 오류 / do.call(function,x) function이 없음

    yhat <- predict(fitted_mdl, newdata = cv.data$predictors, type = "raw")

    score <- rmse(cv.data$y, yhat)

    ans <- list(fitted_mdl = fitted_mdl,
                score = score,
                predictions = data.frame(ID = cv.data$ID, yhat = yhat, y = cv.data$y))

    return(ans)

}

####_Step.3-2_makeOneFoldTestPrediction_train model on one data fold ------------------------
# make prediction from a model fitted to one fold
makeOneFoldTestPrediction <- function(this_fold, feature_set) {
    fitted_mdl <- this_fold$fitted_mdl
    yhat <- predict(fitted_mdl, newdata = feature_set$test$predictors, type = "raw")
    return(yhat)
}

####_Step.4 GBM Model ------------------------
# set caret training parameters
CARET.TRAIN.PARMS <- list(method = "gbm")
CARET.TUNE.GRID <- expand.grid(n.trees = 100, interaction.depth = 10, shrinkage = 0.1, n.minobsinnode = 10)
MODEL.SPECIFIC.PARMS <- list(verbose = 0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method = "none", verboseIter = FALSE, classProbs = FALSE)
CARET.TRAIN.OTHER.PARMS <- list(trControl = CARET.TRAIN.CTRL, tuneGrid = CARET.TUNE.GRID, metric = "RMSE")

# generate features for Level 1
if (!require("plyr")) { install.packages("plyr") }

gbm_set <- llply(data_folds, trainOneFold, L0FeatureSet1)
# L0FeatureSet1 : 59개 변수(확인+시험)
# trainOneFold : traind data를 random fold 적용
# data_folds : train data -> random Fold 5개 list로 저장

# final model fit
gbm_mdl <- do.call(train, c(list(x = L0FeatureSet1$train$predictors, y = L0FeatureSet1$train$y),
                 CARET.TRAIN.PARMS, MODEL.SPECIFIC.PARMS, CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c, lapply(gbm_set, function(x) { x$predictions$y }))
cv_yhat <- do.call(c, lapply(gbm_set, function(x) { x$predictions$yhat }))
rmse(cv_y, cv_yhat)

cat("Average CV rmse:", mean(do.call(c, lapply(gbm_set, function(x) { x$score }))))

test_gbm_yhat <- predict(gbm_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")
gbm_submission <- cbind(Id = L0FeatureSet1$test$id, SalePrice = exp(test_gbm_yhat))
write.csv(gbm_submission, file = "gbm_sumbission.csv", row.names = FALSE)

####_Step.5 xgboost Model ------------------------
# set caret training parameters
CARET.TRAIN.PARMS <- list(method = "xgbTree")

CARET.TUNE.GRID <- expand.grid(nrounds = 800,
                                max_depth = 10,
                                eta = 0.03,
                                gamma = 0.1,
                                colsample_bytree = 0.4,
                                min_child_weight = 1)

MODEL.SPECIFIC.PARMS <- list(verbose = 0) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method = "none",
                                 verboseIter = FALSE,
                                 classProbs = FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl = CARET.TRAIN.CTRL,
                           tuneGrid = CARET.TUNE.GRID,
                           metric = "RMSE")

# generate Level 1 features
xgb_set <- llply(data_folds, trainOneFold, L0FeatureSet2)

# final model fit
xgb_mdl <- do.call(train,
                 c(list(x = L0FeatureSet2$train$predictors, y = L0FeatureSet2$train$y),
                 CARET.TRAIN.PARMS,
                 MODEL.SPECIFIC.PARMS,
                 CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c, lapply(xgb_set, function(x) { x$predictions$y }))
cv_yhat <- do.call(c, lapply(xgb_set, function(x) { x$predictions$yhat }))
rmse(cv_y, cv_yhat)

cat("Average CV rmse:", mean(do.call(c, lapply(xgb_set, function(x) { x$score }))))

# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_xgb_yhat <- predict(xgb_mdl, newdata = L0FeatureSet2$test$predictors, type = "raw")
xgb_submission <- cbind(Id = L0FeatureSet2$test$id, SalePrice = exp(test_xgb_yhat))

write.csv(xgb_submission, file = "xgb_sumbission.csv", row.names = FALSE)

####_Step.6 ranger Model ------------------------
# set caret training parameters
CARET.TRAIN.PARMS <- list(method = "ranger")

CARET.TUNE.GRID <- expand.grid(mtry = 2 * as.integer(sqrt(ncol(L0FeatureSet1$train$predictors))))

MODEL.SPECIFIC.PARMS <- list(verbose = 0, num.trees = 500) #NULL # Other model specific parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method = "none",
                                 verboseIter = FALSE,
                                 classProbs = FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl = CARET.TRAIN.CTRL,
                           tuneGrid = CARET.TUNE.GRID,
                           metric = "RMSE")

# generate Level 1 features
rngr_set <- llply(data_folds, trainOneFold, L0FeatureSet1)

# final model fit
rngr_mdl <- do.call(train,
                 c(list(x = L0FeatureSet1$train$predictors, y = L0FeatureSet1$train$y),
                 CARET.TRAIN.PARMS,
                 MODEL.SPECIFIC.PARMS,
                 CARET.TRAIN.OTHER.PARMS))

# CV Error Estimate
cv_y <- do.call(c, lapply(rngr_set, function(x) { x$predictions$y }))
cv_yhat <- do.call(c, lapply(rngr_set, function(x) { x$predictions$yhat }))
rmse(cv_y, cv_yhat)

cat("Average CV rmse:", mean(do.call(c, lapply(rngr_set, function(x) { x$score }))))

# create test submission.
# A prediction is made by averaging the predictions made by using the models
# fitted for each fold.

test_rngr_yhat <- predict(rngr_mdl, newdata = L0FeatureSet1$test$predictors, type = "raw")
rngr_submission <- cbind(Id = L0FeatureSet1$test$id, SalePrice = exp(test_rngr_yhat))

write.csv(rngr_submission, file = "rngr_sumbission.csv", row.names = FALSE)

####_Step.7 Level 1 Model Training ------------------------
# Create predictions For Level 1 Model
gbm_yhat <- do.call(c, lapply(gbm_set, function(x) { x$predictions$yhat }))
xgb_yhat <- do.call(c, lapply(xgb_set, function(x) { x$predictions$yhat }))
rngr_yhat <- do.call(c, lapply(rngr_set, function(x) { x$predictions$yhat }))

# create Feature Set
L1FeatureSet <- list()

L1FeatureSet$train$id <- do.call(c, lapply(gbm_set, function(x) { x$predictions$ID }))
L1FeatureSet$train$y <- do.call(c, lapply(gbm_set, function(x) { x$predictions$y }))
predictors <- data.frame(gbm_yhat, xgb_yhat, rngr_yhat)
predictors_rank <- t(apply(predictors, 1, rank))
colnames(predictors_rank) <- paste0("rank_", names(predictors))
L1FeatureSet$train$predictors <- predictors #cbind(predictors,predictors_rank)

L1FeatureSet$test$id <- gbm_submission[, "Id"]
L1FeatureSet$test$predictors <- data.frame(gbm_yhat = test_gbm_yhat,
                                      xgb_yhat = test_xgb_yhat,
                                      rngr_yhat = test_rngr_yhat)

# Neural Net Model
# set caret training parameters
CARET.TRAIN.PARMS <- list(method = "nnet")

CARET.TUNE.GRID <- NULL # NULL provides model specific default tuning parameters

# model specific training parameter
CARET.TRAIN.CTRL <- trainControl(method = "repeatedcv",
                                 number = 5,
                                 repeats = 1,
                                 verboseIter = FALSE)

CARET.TRAIN.OTHER.PARMS <- list(trControl = CARET.TRAIN.CTRL,
                            maximize = FALSE,
                           tuneGrid = CARET.TUNE.GRID,
                           tuneLength = 7,
                           metric = "RMSE")

MODEL.SPECIFIC.PARMS <- list(verbose = FALSE, linout = TRUE, trace = FALSE) #NULL # Other model specific parameters


# train the model
set.seed(825)
l1_nnet_mdl <- do.call(train, c(list(x = L1FeatureSet$train$predictors, y = L1FeatureSet$train$y),
                            CARET.TRAIN.PARMS,
                            MODEL.SPECIFIC.PARMS,
                            CARET.TRAIN.OTHER.PARMS))

l1_nnet_mdl

cat("Average CV rmse:", mean(l1_nnet_mdl$resample$RMSE), "\n")

test_l1_nnet_yhat <- predict(l1_nnet_mdl, newdata = L1FeatureSet$test$predictors, type = "raw")
l1_nnet_submission <- cbind(Id = L1FeatureSet$test$id, SalePrice = exp(test_l1_nnet_yhat))
colnames(l1_nnet_submission) <- c("Id", "SalePrice")

write.csv(l1_nnet_submission, file = "l1_nnet_submission.csv", row.names = FALSE)