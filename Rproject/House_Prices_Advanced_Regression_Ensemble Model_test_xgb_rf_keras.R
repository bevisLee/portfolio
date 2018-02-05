

## EDA data import-----------------------

x.train <- read.csv("D:/HousePrice/train_eda.csv")

x.test <- read.csv("D:/HousePrice/test_eda.csv")

x.train <- x.train[, -1]
x.test <- x.test[, -1]

## Data set import -------------
# + 학습에 사용될 알고리즘 : SVM, RandomForest
if (!require("dplyr")) { install.packages("dplyr") }
if (!require("slackr")) { install.packages("slackr") }

SalePrice <- x.train[, 82]

train_df_model <- x.train[, 1:81]
test_df <- x.test
test_id <- c(1461:2919)

train_test <- bind_rows(train_df_model, test_df) ## 합치면서 character 컬럼이 생김
ntrain <- nrow(train_df_model)

features <- names(x.train)

#convert character into factor : character 컬럼 factor 수정
for (f in features) {
    if (is.character(train_test[[f]])) {
        levels = sort(unique(train_test[[f]]))
        train_test[[f]] = as.integer(factor(train_test[[f]], levels = levels))
    }
}

#splitting whole data back again
train_x <- train_test %>% .[1:ntrain,]
test_x <- train_test %>% .[(ntrain + 1):nrow(train_test),]

train_xy <- cbind(train_x, SalePrice)

if (!require("caret")) { install.packages("caret") }
set.seed(222)

inTrain <- createDataPartition(y = train_xy$SalePrice, p = 0.7, list = FALSE)
Training_xy <- train_xy[inTrain,]
Validation_xy <- train_xy[-inTrain,]

# + 학습에 사용될 알고리즘:Keras(tensorflow), XGBoost
## factor에서 numeric 변경 : Keras
train_test_double <- train_test

for (f in features) {
    if (is.factor(train_test_double[[f]])) {
        levels = sort(unique(train_test_double[[f]]))
        train_test_double[[f]] = as.numeric(train_test_double[[f]], levels = levels)
    }
}

#splitting whole data back again
X_train <- train_test_double %>% .[1:ntrain,]
X_test <- train_test_double %>% .[(ntrain + 1):nrow(train_test_double),]

train_XY <- cbind(X_train, SalePrice)

if (!require("caret")) { install.packages("caret") }
set.seed(222)

inTrain_1 <- createDataPartition(y = train_XY$SalePrice, p = 0.7, list = FALSE)
Training_XY <- train_XY[inTrain_1,]
Validation_XY <- train_XY[-inTrain_1,]


## XGboost : RMSE : 21032.65 -> LB 0.12300 -------------
if (!require("xgboost")) { install.packages("xgboost") }
if (!require("Metrics")) { install.packages("Metrics") }

dtrain <- xgb.DMatrix(as.matrix(Training_XY[,-82]), label = Training_XY$SalePrice)
dtest <- xgb.DMatrix(as.matrix(X_test))

##  Hyper Parameters : Build a xgb.DMatrix object
# 참고 - https://datascience.stackexchange.com/questions/9364/hypertuning-xgboost-parameters
# 참고 - https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

slackr_bot("XGboost Hyperparameters Start", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

## 사용자가 grid search 범위 설정 
# -> 범위가 세분화 될 수록 searchGridSubCol 값이 커지므로, 초반에는 범위를 넓게하여 차츰 좁히는 방식 추천 
# searchGridSubCol <- expand.grid(subsample = seq(0.5, 1, by = 0.0001),
#                                colsample_bytree = seq(0.5, 1, by = 0.0001),
#                                ntrees = seq(2000, 3000, by = 100))
#                                min_child_weight = seq(1.5, 2, by = 0.0001),
#                                gamma = seq(0.0, 0.1, by = 0.0001))

searchGridSubCol <- expand.grid(currentSubsampleRate = 0.5213,
                                currentColsampleRate = 0.4603,
                                ntreesNum = 2200,
                                min_child_weight_num = 1.7817,
                                gamma_num = 0.0468)

head(searchGridSubCol)

rmseErrorsHyperparameters_xgb <- data.frame()
for (i in 1:nrow(searchGridSubCol)) {

    #Extract Parameters to test
    currentSubsampleRate <- searchGridSubCol[i,1]
    currentColsampleRate <- searchGridSubCol[i, 2]
    ntreesNum <- searchGridSubCol[i, 3]
    min_child_weight_num <- searchGridSubCol[i, 4]
    gamma_num <- searchGridSubCol[i, 5]

    xgboostModelCV <- xgboost(data = dtrain, nfold = 5, showsd = TRUE,
                             metrics = "rmse", verbose = FALSE, "eval_metric" = "rmse",
                             "objective" = "reg:linear", "max.depth" = 6, "eta" = 0.01,
                             "subsample" = currentSubsampleRate, 
                             "colsample_bytree" = currentColsampleRate,
                             nrounds = ntreesNum, gamma = gamma_num,
                             min_child_weight = min_child_weight_num,
                             nthread = 8, booster = "gbtree")

    ## Predictions
    preds_xgb <- predict(xgboostModelCV, newdata = as.matrix(Validation_XY[, -82]))

    #Save rmse of the last iteration
    rmse <- rmse(Validation_XY$SalePrice, preds2)

    rmseErrorsHyperparameters_xgb_1 <- data.frame(rmse, currentSubsampleRate, currentColsampleRate, ntreesNum, min_child_weight_num, gamma_num)
    rmseErrorsHyperparameters_xgb <- rbind(rmseErrorsHyperparameters_xgb, rmseErrorsHyperparameters_xgb_1) 
}

slackr_bot(rmseErrorsHyperparameters, username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

slackr_bot("XGboost Hyperparameters Finish", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

# subsample : 0.5213 / colsample_bytree : 0.4603 / ntrees : 2200 / min_child_weight : 1.7817 / gamma : 0.0468 -> RMSE : 21032.65    
### prediction---------------------------
currentSubsampleRate <- 0.5213
currentColsampleRate <- 0.4603
ntreesNum <- 2200
min_child_weight_num <- 1.7817
gamma_num <- 0.0468

Dtrain <- xgb.DMatrix(as.matrix(X_train), label = SalePrice)

bst <- xgboost(data = Dtrain, nfold = 5, showsd = TRUE,
               metrics = "rmse", verbose = FALSE, "eval_metric" = "rmse",
               "objective" = "reg:linear", "max.depth" = 6, "eta" = 0.01,
               "subsample" = currentSubsampleRate,
               "colsample_bytree" = currentColsampleRate,
               nrounds = ntreesNum, gamma = gamma_num,
               min_child_weight = min_child_weight_num,
               nthread = 8, booster = "gbtree")

### prediction result save
pred <- data.frame(test_id, xgb_pred <- predict(bst, dtest))
colnames(pred) <- c("Id", "SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/xgb_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)


## LASSO : LB : 0.15485 -------------
if (!require("glmnet")) { install.packages("glmnet") }
if (!require("Metrics")) { install.packages("Metrics") }

set.seed(123)

cv_lasso = cv.glmnet(as.matrix(Training_XY[, -82]), Training_XY$SalePrice, nfolds = 10, type.measure = "mse")

preds_lasso <- predict(cv_lasso, newx = as.matrix(Validation_XY[, -82]), s = "lambda.min")
rmse(Validation_XY$SalePrice, preds_lasso)
# 기본 : 31535.93
# nfold - 1 추가 : 31535.93
# nfold = 5 추가 : 32149.72
# nfold - 10 추가 : 31846.18
# nfold - 10 + type.measure="mse" 추가 : 31237.45

alpha = seq(0, 1, by = 0.00001)

library("progress")
pb <- progress_bar$new(total = length(alpha))

rmseErrorsHyperparameters_lasso <- data.frame()
rmse = 31237.45
for (i in 1:length(alpha)) {
    alpha_num = alpha[i]
    fit <- cv.glmnet(as.matrix(Training_XY[, -82]), Training_XY$SalePrice, alpha = alpha_num, nfolds = 10, type.measure = "mse")
    preds_lasso <- predict(fit, newx = as.matrix(Validation_XY[, -82]), s = "lambda.min")
    rmse1 <- rmse(Validation_XY$SalePrice, preds_lasso)
    rmse = min(rmse, rmse1)

    if (rmse == rmse1) {
        rmseErrorsHyperparameters_lasso_1 <- data.frame(rmse, alpha_num)
        rmseErrorsHyperparameters_lasso <- rbind(rmseErrorsHyperparameters_lasso, rmseErrorsHyperparameters_lasso_1)
        print(rmseErrorsHyperparameters_lasso_1)
    }
    pb$tick()
}
## result : alpha 0 ~ 1 / 0.00001 차이
# rmse alpha_num
# 29961.57         0
# 29771.29     1e-05
# 29089.44     3e-05
# 29038.22   0.00012
# and now run glmnet once more with it
# best alpha = 0.00012
### prediction---------------------------
alpha_num = 0.00012
cv_lasso = cv.glmnet(as.matrix(X_train), SalePrice, alpha = alpha_num, nfolds = 10, type.measure = "mse")

### prediction result save
pred <- data.frame(test_id, lasso_pred <- predict(cv_lasso, newx = as.matrix(X_test),
    s = "lambda.min"))
colnames(pred) <- c("Id", "SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/lasso_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)


## GBM : RMSE : 25048.87 -> LB : 0.12840 -------------
if (!require("iterators")) { install.packages("iterators") }
if (!require("parallel")) { install.packages("parallel") }

##  Hyper Parameters : Build a GBM object
# 참고 - https://www.kaggle.com/aniruddhachakraborty/lasso-gbm-xgboost-top-20-0-12039-using-r
# 참고 - https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/

slackr_bot("GBM Hyperparameters Start", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

## Set up caret model training parameters
CARET.TRAIN.CTRL <- trainControl(method = "repeatedcv", number = 5, repeats = 5,
    verboseIter = FALSE, allowParallel = TRUE)

gbmFit <- train(SalePrice ~ ., method = "gbm", metric = "RMSE", maximize = FALSE,
                trControl = CARET.TRAIN.CTRL,
                tuneGrid = expand.grid(n.trees = c(350, seq(2000, 2500, by = 100)),
                interaction.depth = c(5), shrinkage = c(0.05), n.minobsinnode = c(10)),
                data = Training_xy, verbose = FALSE)

rmseErrorsHyperparameters_gbm <- data.frame(gbmFit$results)
rmseErrorsHyperparameters_gbm_best <- gbmFit$bestTune

rmseErrorsHyperparameters_gbm_best

preds_gbm <- predict(gbmFit, newdata = Validation_xy)
rmse(Validation_xy$SalePrice, preds1)

slackr_bot(rmseErrorsHyperparameters_gbm, username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

slackr_bot("GBM Hyperparameters Finish", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

# n.trees : 2000 / interaction.depth : 5 / shrinkage : 0.05 / n.minobsinnode : 10 -> RMSE : 25048.87    
### prediction---------------------------
gbmfit <- train(SalePrice ~ ., method = "gbm", metric = "RMSE", maximize = FALSE,
             trControl = CARET.TRAIN.CTRL,
             tuneGrid = expand.grid(n.trees = c(2000), interaction.depth = c(5), shrinkage = c(0.05), n.minobsinnode = c(10)),
                data = train_xy, verbose = FALSE)

### prediction result save
pred <- data.frame(test_id, gbm_pred <- predict(gbmfit, test_x))
colnames(pred) <- c("Id", "SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/gbm_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)


## RandomForest : RMSE : 25657.48 -> LB 0.14595 -------------
if (!require("randomForest")) { install.packages("randomForest") }
if (!require("caret")) { install.packages("caret") }

##  Hyper Parameters : Build a RandomForest object
# 참고 - https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/
control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")

tunegrid <- expand.grid(.mtry = c(31)) # 1:15 

##### function
slackr_bot("RandomForest Hyperparameters Start", username = "slackr", channel = "#rf_hyper", incoming_webhook_url = "https://hooks.slack.com/services/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")

# ntree 100:2500
rmseErrorsHyperparameters_rf <- list()
for (ntree in seq(1000, 1200, by = 100)) {
    print(ntree)
    slackr_bot(ntree, username = "slackr", channel = "#rf_hyper", incoming_webhook_url = "https://hooks.slack.com/services/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")
    set.seed(7)
    fit <- train(SalePrice ~ ., data = Training_xy, method = "rf",
                 tuneGrid = tunegrid, trControl = control, ntree = ntree)
    key <- toString(ntree)
    rmseErrorsHyperparameters_rf[[key]] <- fit

    preds_rf <- predict(fit, newdata = Validation_xy)
    rmse <- rmse(Validation_xy$SalePrice, preds_rf)
    print(cbind(ntree,rmse))
}

slackr_bot(rmseErrorsHyperparameters_rf, username = "slackr", channel = "#rf_hyper", incoming_webhook_url = "https://hooks.slack.com/services/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")

slackr_bot("RandomForest Hyperparameters Finish", username = "slackr", channel = "#rf_hyper", incoming_webhook_url = "https://hooks.slack.com/services/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")

## 결과 저장
rmseErrorsHyperparameters_rf_result <- data.frame()

ntree <- 2000
a <- rmseErrorsHyperparameters_rf$`2000`$results[4,]
rmseErrorsHyperparameters_rf_result_1 <- cbind(ntree,a)
rmseErrorsHyperparameters_rf_result <- rbind(rmseErrorsHyperparameters_rf_result, rmseErrorsHyperparameters_rf_result_1)

rmseErrorsHyperparameters_rf_result
### Best gride search parameters ---------------------------
# mtry : 5 / ntree 500 -> RMSE : 33889.03 
# mtry : 29 / ntree 1200 -> RMSE : 28876.83
# mtry : 31 / ntree 1000 -> RMSE : 25657.48
### prediction ---------------------------
tunegrid <- expand.grid(.mtry = 31)
ntree = 1000

rf_model <- train(SalePrice ~ ., data = train_xy, method = "rf",
                 tuneGrid = tunegrid, trControl = control, ntree = ntree)

pred <- data.frame(test_id, rf_pred <- predict(rf_model, test_x))
colnames(pred) <- c("Id","SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/rf_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)
# mtry : 5 / ntree : 500 / rmse : 33889.03 -> LB : 0.16743
# mtry : 29 / ntree : 1200 / rmse : 28184.84 -> LB : 0.14645
# mtry : 31 / ntree : 1000 / rmse : 25657.48 -> LB : 0.14595


## Keras : LB 0.17305 -------------    
if (!require("tensorflow")) { install.packages("tensorflow") }
if (!require("keras")) { install.packages("keras") }

##  Hyper Parameters : Build a Keras object
# 참고 - https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

# keras 앙상블 참고 - https://blog.rescale.com/deep-neural-network-hyper-parameter-optimization/

### define the grid search parameters
# c("relu", "elu", "selu", "hard_sigmoid", "linear", "sigmoid", "softmax", "softplus", "softsign", "tanh")

input_activation = c("selu", "elu", "relu", "softplus", "linear","sigmoid")
hidden1_activation = c("selu", "elu", "relu", "softplus", "linear", "sigmoid")
hidden2_activation = c("selu", "elu", "relu", "softplus", "linear", "sigmoid")
output_activation = c("selu", "elu", "relu", "softplus", "linear", "sigmoid")

# input_dropout_rate = c(0.1, 0.5, 0.9) # seq(0.0, 0.9, by = 0.1)
# hidden1_dropout_rate = c(0.5, 0.9) # seq(0.0, 0.9, by = 0.1)

epochs_num = 200 # seq(200, 600, by = 10)
batch_size_num = 10 # seq(800, 1500, by = 100)

# units 
input_num_units = ncol(X_train) # input
# hidden_num_units1 = 1000 # hidden
output_num_units = 1 # output

#history
verbose_num = 1
validation_split_num = 0.1

# define the grid search parameters
grid_search_keras = expand.grid(input_activation = input_activation,
                                hidden1_activation = hidden1_activation,
                                hidden2_activation = hidden2_activation,
                                output_activation = output_activation,
#                               input_dropout_rate = input_dropout_rate, # dropout_rate
#                               hidden1_dropout_rate = hidden1_dropout_rate,
                                epochs_num = epochs_num, # epochs
                                batch_size_num = batch_size_num) # batch_size


head(grid_search_keras)
slackr_bot(head(grid_search_keras), username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

## create_model function
slackr_bot("Keras Hyperparameters Start", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

rmseErrorsHyperparameters_keras <- data.frame()
tensorboard(log_dir = "logs")

rmse = 38777.2312387326
for (i in 392:nrow(grid_search_keras)) {
    set.seed(1234)
    rm(model, history)

    #Extract Parameters to test
    input_activation <- as.character(grid_search_keras[i, 1])
    hidden1_activation <- as.character(grid_search_keras[i, 2])
    hidden2_activation <- as.character(grid_search_keras[i, 3])
    output_activation <- as.character(grid_search_keras[i, 4])
    epochs_num <- as.integer(grid_search_keras[i, 5])
    batch_size_num <- as.integer(grid_search_keras[i, 6])

    model <- keras_model_sequential()
    Sys.sleep(2)

    model %>%
    layer_dense(units = input_num_units, input_shape = c(ncol(X_train)), activation = input_activation, kernel_initializer = 'normal') %>%
    #    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 2000, activation = hidden1_activation, kernel_initializer = 'normal') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 2000, activation = hidden2_activation, kernel_initializer = 'normal') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1, activation = output_activation, kernel_initializer = 'normal')

    model %>%
    compile(loss = 'mse', optimizer = 'adam', metrics = 'accuracy')

    train.ind <- sample(1:nrow(X_train), 0.7 * nrow(X_train))

    summary(model)

    history <- model %>%
                fit(as.matrix(X_train[train.ind,]), as.matrix(train_XY[train.ind, 82]),
                epochs = epochs_num, batch_size = batch_size_num, verbose = verbose_num,
                validation_split = validation_split_num,
                callback_early_stopping(monitor = "val_loss", mode = "min")) # min_delta = 0, patience = 0, verbose = 0

    score <- model %>% evaluate(as.matrix(X_train[-train.ind,]), train_XY[-train.ind, 82])
    sqrt(score$loss) #rmse

    rmse1 <- sqrt(score$loss) #rmse
    rmse = min(rmse,rmse1)

    key <- toString(i)
    rmseErrorsHyperparameters_keras_1 <- data.frame(i, rmse1, input_activation, hidden1_activation, hidden2_activation, output_activation, batch_size_num, epochs_num)
    colnames(rmseErrorsHyperparameters_keras_1) <- c("i", "rmse", "input_activation", "hidden1_activation", "hidden2_activation", "output_activation", "batch_size_num", "epochs_num")
    rmseErrorsHyperparameters_keras <- rbind(rmseErrorsHyperparameters_keras, rmseErrorsHyperparameters_keras_1)

    print(grid_search_keras[i,])
    print(rmse1)

    if (rmse == rmse1) {
        slackr_bot(grid_search_keras[i,], username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")
        slackr_bot(c(i, rmse), username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

        epochs_num = 1000

        history <- model %>%
                fit(as.matrix(X_train[train.ind,]), as.matrix(train_XY[train.ind, 82]),
                epochs = epochs_num, batch_size = batch_size_num, verbose = verbose_num,
                validation_split = validation_split_num,
                callbacks = callback_tensorboard(paste0("logs/run_",i,sep=" ")))

        pred_keras <- data.frame(ID = test_id, SalePrice = predict(model, as.matrix(X_test)))
        write.csv(pred_keras, paste0("D:/HousePrice/keras_Hyper_Parameters_",i,"_",format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)
    }
    write.csv(rmseErrorsHyperparameters_keras,"D:/HousePrice/rmseErrorsHyperparameters_keras.csv")
}

slackr_bot(rmseErrorsHyperparameters_keras, username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

slackr_bot("Keras Hyperparameters Finish", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")
### Best grid search parameters---------------------------
# result : rmseErrorsHyperparameters_keras
#  i   rmse    input_activation hidden1_activation hidden2_activation output_activation batch_size_num epochs_num
#  1 32891.16       relu            sigmoid             relu            relu             800             3000
#  2 33558.31       relu            sigmoid             relu            relu             800             5000
#  3 34514.03       softplus        sigmoid             elu             relu             800             1000
#  4 33948.03       softplus        sigmoid             elu             relu             800             2000
#  5 32991.05       softplus        sigmoid             elu             relu             800             5000
#  6 36068.37       softplus        sigmoid             elu             relu             5000            1000
#  7 36330.29       softplus        sigmoid             elu             relu             100             1000
#  8 32603.07       softplus        sigmoid             elu             relu             10              1000
#  9 31763.23       elu             sigmoid             selu            relu             10              1000
# 10 38107.49       elu             sigmoid             selu            relu             10              5000
# 11 00000.00       softplus        elu                 softplus        selu             10              1000

input_activation = "softplus"
hidden1_activation = "elu"
hidden2_activation = "softplus"
output_activation = "selu"

epochs_num = 1000 
batch_size_num = 10 

# units 
input_num_units = ncol(X_train) # input
output_num_units = 1 # output

#history
verbose_num = 1
validation_split_num = 0.1

model %>%
    layer_dense(units = input_num_units, input_shape = c(ncol(X_train)), activation = input_activation, kernel_initializer = 'normal') %>%
#    layer_dropout(rate = 0.5) %>%
layer_dense(units = 2000, activation = hidden1_activation, kernel_initializer = 'normal') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 2000, activation = hidden2_activation, kernel_initializer = 'normal') %>%
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 1, activation = output_activation, kernel_initializer = 'normal')

model %>%
    compile(loss = 'mse', optimizer = 'adam', metrics = 'accuracy')

train.ind <- sample(1:nrow(X_train), 0.7 * nrow(X_train))

summary(model)

history <- model %>%
                fit(as.matrix(X_train[train.ind,]), as.matrix(train_XY[train.ind, 82]), epochs = epochs_num, batch_size = batch_size_num, verbose = verbose_num, validation_split = validation_split_num)

score <- model %>% evaluate(as.matrix(X_train[-train.ind,]), train_XY[-train.ind, 82])
sqrt(score$loss) #rmse
# batch : 800 / epoch : 50 / input_activation : selu / hidden1_activation : relu / output_activation : linear / drop : 0.0 / drop : 0.0 -> rmse : 43424.25876 , LB : 0.26357
### prediction---------------------------
pred_keras <- data.frame(ID = test_id, SalePrice = predict(model, as.matrix(X_test)))

write.csv(pred_keras, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/keras_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)


## Ensemble Prediction : LB 0.12249  ----------------------------------
# RMSE score for Simple Average of the three models
rmse(Validation_xy$SalePrice, (preds_xgb + preds_gbm + preds_rf + preds_keras + preds_lasso) / 5)
rmse(Validation_xy$SalePrice, (preds_xgb + preds_gbm + preds_lasso) / 3)

# RMSE score for Weighted Average of the All models
RMSE_Weighted <- expand.grid(xgb_num = seq(0.1, 0.99, by = 0.01),
                             gbm_num = seq(0.1, 0.99, by = 0.01),
                             rf_num = seq(0.1, 0.99, by = 0.01),
                             keras_num = seq(0.1, 0.99, by = 0.01),
                             lasso_num = seq(0.1, 0.99, by = 0.01))

RMSE_Weighted <- expand.grid(xgb_num = seq(0.1, 0.99, by = 0.01),
                             gbm_num = seq(0.1, 0.99, by = 0.01),
                             lasso_num = seq(0.1, 0.99, by = 0.01))

sum <- rowSums(RMSE_Weighted)
RMSE_Weighted <- cbind(RMSE_Weighted,sum)

RMSE_Weighted <- RMSE_Weighted[RMSE_Weighted$sum == 1.0,]

RMSE_Weighted_score <- data.frame()
for (i in 1:nrow(RMSE_Weighted)) {
    xgb_num <- RMSE_Weighted[i, 1]
    gbm_num <- RMSE_Weighted[i, 2]
    lasso_num <- RMSE_Weighted[i, 3]
#    rf_num <- RMSE_Weighted[i, 4]
#    keras_num <- RMSE_Weighted[i, 5]
    
    rmse <- rmse(Validation_xy$SalePrice, (xgb_num * preds_gbm + gbm_num * preds_xgb + lasso_num * preds_lasso)) # rf_num * preds_rf + keras_num * preds_keras))

    RMSE_Weighted_score_1 <- data.frame(rmse, xgb_num, gbm_num, lasso_num) # rf_num, keras_num)
    RMSE_Weighted_score <- rbind(RMSE_Weighted_score, RMSE_Weighted_score_1)
}
# result  ----------------------------------
# Weighted 보다는 LB Score에 따라 가중치가 더 효과적
xgb_num <- 0.64
gbm_num <- 0.36

rmse <- rmse(Validation_xy$SalePrice, (xgb_num * preds_gbm + gbm_num * preds_xgb + lasso_num * preds_lasso))
rmse

pred <- data.frame(Id = test_id,
                   ProbabilityXGB = xgb_pred,
                   ProbabilityGBM = gbm_pred)
#                   ProbabilityLASSO = lasso_pred)
#                   ProbabilityRF = rf_pred,
#                  ProbabilityKeras = keras_pred)

colnames(pred) <- c("Id", "ProbabilityXGB", "ProbabilityGBM", "ProbabilityLASSO")

# sigmoid function to convert to probability
# GBM X : pred$ProbabilityGBM <- 1 / (1 + exp(-pred$ProbabilityGBM)) 

# After submitting each model individually, this upweights models with greater individual AUC
# pred$Probability <- ((pred$ProbabilitySVM) + (pred$ProbabilityRF) * 20 + (pred$ProbabilityXGB) * 14) / 35

pred$Probability <- ((pred$ProbabilityXGB) * xgb_num + (pred$ProbabilityGBM) * gbm_num)

pred <- pred %>% dplyr::select(Id, Probability)
colnames(pred) <- c("Id", "SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/Ensemble_model", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)

## 앙상블 참고 - https://github.com/h2oai/h2o-3/blob/master/h2o-r/ensemble/h2oEnsemble-package/R/ensemble.R
## https://github.com/h2oai/h2o-3/tree/master/h2o-r/ensemble/h2oEnsemble-package/R


## Support Vector Machines : LB 0.43200 -> 수정 필요 -------------

##  Hyper Parameters : Build a Support Vector Machines object
# 참고 - http://blog.revolutionanalytics.com/2016/06/bayesian-optimization-of-machine-learning-models.html

rand_ctrl <- trainControl(method = "repeatedcv", repeats = 5, search = "random")

set.seed(308)
rand_search <- train(SalePrice ~ ., data = train_xy, method = "svmRadial",
## Create 20 random parameter values
                    tuneLength = 20,
                    metric = "RMSE",
                    preProc = c("center", "scale"),
                    trControl = rand_ctrl)
#     sigma         C             RMSE      Rsquared     MAE     
#   0.0009442604    0.20456054  80931.69  0.002853269  55529.44

rand_search

## Define the resampling method
ctrl <- trainControl(method = "repeatedcv", repeats = 5)

## Use this function to optimize the model. The two parameters are 
## evaluated on the log scale given their range and scope. 
svm_fit_bayes <- function(logC, logSigma) {
    ## Use the same model code but for a single (C, sigma) pair. 
    txt <- capture.output(
         mod <- train(SalePrice ~ ., data = train_xy,
         method = "svmRadial",
        preProc = c("center", "scale"),
        metric = "RMSE",
        trControl = ctrl,
        tuneGrid = data.frame(C = exp(logC), sigma = exp(logSigma)))
     )
    ## The function wants to _maximize_ the outcome so we return 
    ## the negative of the resampled RMSE value. `Pred` can be used
    ## to return predicted values but we'll avoid that and use zero
    list(Score = -getTrainPerf(mod)[, "TrainRMSE"], Pred = 0)
}

## Define the bounds of the search. 
lower_bounds <- c(logC = -20, logSigma = -9)
upper_bounds <- c(logC = 20, logSigma = 9)
bounds <- list(logC = c(lower_bounds[1], upper_bounds[1]),
               logSigma = c(lower_bounds[2], upper_bounds[2]))

## Create a grid of values as the input into the BO code
initial_grid <- rand_search$results[, c("C", "sigma", "RMSE")]
initial_grid$C <- log(initial_grid$C)
initial_grid$sigma <- log(initial_grid$sigma)
initial_grid$RMSE <- -initial_grid$RMSE
names(initial_grid) <- c("logC", "logSigma", "Value")

## Run the optimization with the initial grid and do
## 30 iterations. We will choose new parameter values
## using the upper confidence bound using 1 std. dev. 

if (!require("rBayesianOptimization")) { install.packages("rBayesianOptimization") }

slackr_bot("Support Vector Machines Hyperparameters Start", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

set.seed(8606)
ba_search <- BayesianOptimization(svm_fit_bayes,
                                  bounds = bounds,
                                  init_grid_dt = initial_grid,
                                  init_points = 0,
                                  n_iter = 30,
                                  acq = "ucb",
                                  kappa = 1,
                                  eps = 0.0,
                                  verbose = TRUE)

ba_search
# logC : 20 / logSigma : 9 / Value : -78907.74

slackr_bot(ba_search, username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

slackr_bot("Support Vector Machines Hyperparameters Finish", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

set.seed(308)
final_search <- train(SalePrice ~ ., data = train_xy, method = "svmRadial",
                      tuneGrid = data.frame(C = exp(ba_search$Best_Par["logC"]),
                      sigma = exp(ba_search$Best_Par["logSigma"])),
                      metric = "RMSE", preProc = c("center", "scale"),
                      trControl = ctrl)

# 처음 실행한 SVM 결과와 최종 최적화한 결과 비교
compare_models(final_search, rand_search)
# final_search
#     RMSE  Rsquared      MAE
# 78974.55 0.002984776 57707.73

# rand_search
#    RMSE      Rsquared     MAE     
# 80931.69  0.002853269  55529.44

# 처음 실행한 SVM 결과 -> RMSE :79523.46 vs 최종 최적화한 SVM 결과 -> RSME : 79523.46
postResample(predict(rand_search, test_x), train_xy$SalePrice)
postResample(predict(final_search, test_x), train_xy$SalePrice)
### prediction ---------------------------
pred <- data.frame(test_id, svm_pred <- predict(final_search, test_x))
colnames(pred) <- c("Id", "SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/svm_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)


## GBM :  -------------
if (!require("gbm")) { install.packages("gbm") }
if (!require("Metrics")) { install.packages("Metrics") }

searchGridSubCol <- expand.grid(ntreesNum = seq(500, 3000, by = 100),
                                shrinkage_num = 0.01, # seq(0.001, 0.01, by = 0.001)
                                n.minobsinnode_num = seq(20, 10, by = -1),
                                interaction.depth_num = seq(49, 1, by = -1)) # interaction.depth must be less than 50

head(searchGridSubCol)

rmseErrorsHyperparameters_gbm <- data.frame()
rmse = 24000
for (i in 1:nrow(searchGridSubCol)) {

    #Extract Parameters to test
    ntreesNum <- searchGridSubCol[i, 1]
    shrinkage_num <- searchGridSubCol[i, 2]
    n.minobsinnode_num <- searchGridSubCol[i, 3]
    interaction.depth_num <- searchGridSubCol[i, 4]

    gbmModel = gbm(formula = SalePrice ~ .,
                   data = Training_xy,
                   n.trees = ntreesNum, # number of trees
                   shrinkage = shrinkage_num, # shrinkage or learning rate, 0.001 to 0.1 usually work
                   n.minobsinnode = n.minobsinnode_num, # minimum total weight needed in each node
                   interaction.depth = interaction.depth_num, # 1: additive model, 2: two-way interactions, etc.
                   cv.folds = 5, # 5-fold cross-validation
                   distribution = "gaussian")

    ## Predictions
    preds_xgb = predict(object = gbmModel, newdata = Validation_xy, n.trees = ntreesNum)

    #Save rmse of the last iteration
    rmse1 <- rmse(Validation_xy$SalePrice, preds_xgb)
    rmse <- min(rmse, rmse1)

    print(i)
    print(rmse1)

    if (rmse == rmse1) {
        rmseErrorsHyperparameters_gbm_1 <- data.frame(i, rmse1, ntreesNum, shrinkage_num, n.minobsinnode_num, interaction.depth_num)
        colnames(rmseErrorsHyperparameters_gbm_1) <- c("i", "rmse", "ntreesNum", "shrinkage_num", "n.minobsinnode_num", "interaction.depth_num")
        rmseErrorsHyperparameters_gbm <- rbind(rmseErrorsHyperparameters_gbm, rmseErrorsHyperparameters_gbm_1)
        print(rmseErrorsHyperparameters_gbm_1)
    }
}

tail(rmseErrorsHyperparameters_gbm, 5)

# i rmse ntreesNum shrinkage_num n.minobsinnode_num
# 90 28401.69 1000 0.009 5
# 590 28350.19 1000 0.009 10
# 600 28288.01 1000 0.010 10
# 890 28224.46 1000 0.009 13
# 899 28167.20 900 0.010 13
# 990 28050.53 1000 0.009 14
# 1090 27911.20 1000 0.009 15
# 1100 27709.37 1000 0.010 15
# 1200 27610.62 1000 0.010 16
# 1400 27600.15 1000 0.010 18
# 1600 27420.64 1000 0.010 20
# 5 23916.86 900 0.01 20 49
# 7 23873.78 1100 0.01 20 49
# 8 23813.89 1200 0.01 20 49
# 11 23540.49 1500 0.01 20 49
# 35 23530.84 1300 0.01 19 49
# 323 23518.88 1500 0.01 19 48
# 1178 23512.10 1200 0.01 19 45

# i rmse ntreesNum shrinkage_num n.minobsinnode_num interaction.depth_num
# 12 23376.01 1600 0.01 20 10

ntreesNum <- searchGridSubCol[tail(rmseErrorsHyperparameters_gbm$i, 1), 1]
shrinkage_num <- searchGridSubCol[tail(rmseErrorsHyperparameters_gbm$i, 1), 2]
n.minobsinnode_num <- searchGridSubCol[tail(rmseErrorsHyperparameters_gbm$i, 1), 3]

gbmModel = gbm(formula = SalePrice ~ .,
               data = train_xy,
               n.trees = ntreesNum, # number of trees
               shrinkage = shrinkage_num, # shrinkage or learning rate, 0.001 to 0.1 usually work
               n.minobsinnode = n.minobsinnode_num, # minimum total weight needed in each node
               cv.folds = 5, # 5-fold cross-validation
               distribution = "gaussian",
               verbose = TRUE)

### prediction_train all result save
pred <- data.frame(test_id, gbm_pred <- predict(object = gbmModel, newdata = test_x, n.trees = ntreesNum))
colnames(pred) <- c("Id", "SalePrice")

write.csv(pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/gbm_Hyper_Parameters", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)
