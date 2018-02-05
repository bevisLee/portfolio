

## 1_data EDA ----------------------------------------
# 참고 - http://hamelg.blogspot.kr/2016/09/kaggle-home-price-prediction-tutorial.html

train = read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train.csv")
test = read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test.csv")

dim(train)
dim(test)

str(train)
str(test)

levels(train$MiscFeature)
levels(test$MiscFeature)

train = read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train.csv", stringsAsFactors = FALSE)
test = read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test.csv", stringsAsFactors = FALSE)

# Remove the target variable not found in test set
SalePrice = train$SalePrice
train$SalePrice = NULL

# Combine data sets
full_data = rbind(train, test)

# Convert character columns to factor, filling NA values with "missing"
for (col in colnames(full_data)) {
    if (typeof(full_data[, col]) == "character") {
        new_col = full_data[, col]
        new_col[is.na(new_col)] = "missing"
        full_data[col] = as.factor(new_col)
    }
}

# Separate out our train and test sets
train = full_data[1:nrow(train),]
train$SalePrice = SalePrice
test = full_data[(nrow(train) + 1):nrow(full_data),]

summary(train)

# Fill remaining NA values with -1
train[is.na(train)] = -1
test[is.na(test)] = -1

for (col in colnames(train)) {
    if (is.numeric(train[, col])) {
        if (abs(cor(train[, col], train$SalePrice)) > 0.5) {
            print(col)
            print(cor(train[, col], train$SalePrice))
        }
    }
}

for (col in colnames(train)) {
    if (is.numeric(train[, col])) {
        if (abs(cor(train[, col], train$SalePrice)) < 0.1) {
            print(col)
            print(cor(train[, col], train$SalePrice))
        }
    }
}

cors = cor(train[, sapply(train, is.numeric)])
high_cor = which(abs(cors) > 0.6 & (abs(cors) < 1))
rows = rownames(cors)[((high_cor - 1) %/% 38) + 1]
cols = colnames(cors)[ifelse(high_cor %% 38 == 0, 38, high_cor %% 38)]
vals = cors[high_cor]

cor_data = data.frame(cols = cols, rows = rows, correlation = vals)
cor_data

for (col in colnames(train)) {
    if (is.numeric(train[, col])) {
        plot(density(train[, col]), main = col)
    }
}

# Add variable that combines above grade living area with basement sq footage
train$total_sq_footage = train$GrLivArea + train$TotalBsmtSF
test$total_sq_footage = test$GrLivArea + test$TotalBsmtSF

# Add variable that combines above ground and basement full and half baths
train$total_baths = train$BsmtFullBath + train$FullBath + (0.5 * (train$BsmtHalfBath + train$HalfBath))
test$total_baths = test$BsmtFullBath + test$FullBath + (0.5 * (test$BsmtHalfBath + test$HalfBath))

# Remove Id since it should have no value in prediction
train$Id = NULL
test$Id = NULL

sapply(train[, 1:82], function(x) sum(is.na(x)))

test_id <- full_data[1461:2919, 1]

rm(col, cols, cor_data, cors, high_cor, new_col, rows, vals, SalePrice,full_data)

## 2._Ensemble learning -----------------
require("h2oEnsemble")
require("slackr")

# h2o 클러스터 시작
h2o.init(startH2O = TRUE, nthreads = -1, max_mem_size = "91G") # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # (Optional) Remove all objects in H2O cluster

# h2o 인스턴트 상태 확인
h2o.init()

train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)

colnames(train.h2o)

#dependent variable (SalePrice)
y.dep <- 80

#independent variables (dropping ID variables)
x.indep <- c(1:79, 81:82)

### Here is an example of how to generate a custom learner wrappers:
# glm
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)

# rf
h2o.randomForest.1 <- function(..., ntrees = ntree, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = ntree, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = ntree, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = ntree, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)

# gbm
h2o.gbm.1 <- function(..., ntrees = ntree, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = ntree, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = ntree, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = ntree, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = ntree, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = ntree, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = ntree, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = ntree, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)

# deeplearinig
h2o.deeplearning.1 <- function(..., hidden = c(500, 500), activation = "Rectifier", epochs = 50, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200, 200, 200), activation = "Tanh", epochs = 50, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500, 500), activation = "RectifierWithDropout", epochs = 50, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500, 500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100, 100, 100), activation = "Rectifier", epochs = 50, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50, 50), activation = "Rectifier", epochs = 50, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100, 100), activation = "Rectifier", epochs = 50, seed = 1) h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

ntree = 800

### Specify the base learner library & the metalearner
## learner all
# learner <- c("h2o.randomForest.wrapper", "h2o.gbm.wrapper", "h2o.deeplearning.wrapper", "h2o.glm.wrapper", "h2o.glm.1", "h2o.glm.2", "h2o.glm.3", "h2o.randomForest.1", "h2o.randomForest.2", "h2o.randomForest.3", "h2o.randomForest.4", "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3", "h2o.gbm.4", "h2o.gbm.5", "h2o.gbm.6", "h2o.gbm.7", "h2o.gbm.8", "h2o.deeplearning.1", "h2o.deeplearning.2", "h2o.deeplearning.3", "h2o.deeplearning.4", "h2o.deeplearning.5", "h2o.deeplearning.6", "h2o.deeplearning.7")

## learner best score
learner <- c("h2o.deeplearning.wrapper", "h2o.randomForest.2", "h2o.randomForest.3") 

metalearner <- learner[1]
family <- "AUTO"
cvControl <- list(V = 18, shuffle = TRUE)

rm(pred, fit)

slackr_bot("h2o Ensemble learning start", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

fit <- h2o.ensemble(x = x.indep, y = y.dep,
                    training_frame = train.h2o,
                    family = family,
                    learner = learner,
                    metalearner = metalearner,
                    cvControl = cvControl)

slackr_bot(c(learner, metalearner, family, cvControl), username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

pred <- predict.h2o.ensemble(fit, test.h2o)

slackr_bot(pred, username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

h2o_Ensemble <- data.frame(Id = test_id, SalePrice = as.data.frame(pred$pred))
colnames(h2o_Ensemble) <- c("Id", "SalePrice")

write.csv(h2o_Ensemble, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/h2o_Ensemble_", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)

slackr_bot("h2o Ensemble learning finish", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

pred

h2o.shutdown(prompt = FALSE)


# custom learner test-----------------------------
## custom learner result

learner_1 <- c("h2o.randomForest.wrapper", "h2o.gbm.wrapper", "h2o.deeplearning.wrapper", "h2o.glm.wrapper", "h2o.glm.1", "h2o.glm.2", "h2o.glm.3", "h2o.randomForest.1", "h2o.randomForest.2", "h2o.randomForest.3", "h2o.randomForest.4", "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3", "h2o.gbm.4", "h2o.gbm.5", "h2o.gbm.6", "h2o.gbm.7", "h2o.gbm.8", "h2o.deeplearning.1", "h2o.deeplearning.2", "h2o.deeplearning.3", "h2o.deeplearning.4", "h2o.deeplearning.5", "h2o.deeplearning.6", "h2o.deeplearning.7")

ntree = 800

for (i in 23:length(learner_1)) {
    require("h2oEnsemble")
    require("slackr")

    # h2o 클러스터 시작
    h2o.init(startH2O = TRUE, nthreads = -1, max_mem_size = "91G") # Start an H2O cluster with nthreads = num cores on your machine
    h2o.removeAll() # (Optional) Remove all objects in H2O cluster

    # h2o 인스턴트 상태 확인
    h2o.init()

    train.h2o <- as.h2o(train)
    test.h2o <- as.h2o(test)

    colnames(train.h2o)

    #dependent variable (SalePrice)
    y.dep <- 80

    #independent variables (dropping ID variables)
    x.indep <- c(1:79, 81:82)

    learner <- c(learner_1[i])
    metalearner <- learner[1]
    family <- "AUTO"
    cvControl <- list(V = 18, shuffle = TRUE)

    print(paste0("custom learning..._", i, "번째 실행중_", learner))
    slackr_bot(paste0("custom learning..._", i, "번째 실행중_", learner), username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")


    rm(pred, fit)

    fit <- h2o.ensemble(x = x.indep, y = y.dep,
                        training_frame = train.h2o,
                        family = family,
                        learner = learner,
                        metalearner = metalearner,
                        cvControl = cvControl)

    pred <- predict.h2o.ensemble(fit, test.h2o)
    custom_result <- cbind(custom_result, as.data.frame(pred$basepred))

    h2o.shutdown(prompt = FALSE)
    Sys.sleep(20)
}


custom_result <- as.data.frame(pred$basepred)
write.csv(custom_result, 'C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/custom_result.csv')

# custom learner test-----------------------------
## score
# h2o.randomForest.wrapper : -80,111,345 
# h2o.gbm.wrapper : -81,185,093 
# h2o.deeplearning.wrapper : -79,627,562 
# h2o.glm.wrapper : -83,144,545 
# h2o.glm.1 : -83,143,273 
# h2o.glm.2 : -83,144,545 
# h2o.glm.3 : -80,204,955 
# h2o.randomForest.1 : -79,840,549 
# h2o.randomForest.2 : -79,787,785 
# h2o.randomForest.3 : -79,831,744 
# h2o.randomForest.4 : -79,840,549 
# h2o.gbm.1 : -81,391,817 
# h2o.gbm.2 : -81,057,781 
# h2o.gbm.3 : -80,759,512 
# h2o.gbm.4 : -80,787,244 
# h2o.gbm.5 : -81,408,376 
# h2o.gbm.6 : -80,309,034 
# h2o.gbm.7 : -81,391,817 
# h2o.gbm.8 : -80,648,441 
# h2o.deeplearning.1 : -90,245,332 
# h2o.deeplearning.2 : -91,141,572 
# h2o.deeplearning.3 : -90,417,624 


## 2-1_Ensemble h2o Test--------------------
require("h2oEnsemble")
require("slackr")

# h2o 클러스터 시작
h2o.init(nthreads = -1) # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # (Optional) Remove all objects in H2O cluster

# write.csv(train, "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train_eda.csv")
# write.csv(test, "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test_eda.csv")

#### Load Data into H2O Cluster
x.train <- h2o.importFile(path = normalizePath("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train_eda.csv"))
x.test <- h2o.importFile(path = normalizePath("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test_eda.csv"))

train_Id <- train$Id
test_Id <- test$Id

train <- train[, -1] #Id 제외
train <- train[, -1] #Id 제외

y <- "SalePrice"
x <- setdiff(names(train), y)
#For binary classification, the response should be encoded as factor (also known as the [enum](https://docs.oracle.com/javase/tutorial/java/javaOO/enum.html) type in Java).  The user can specify column types in the `h2o.importFile` command, or you can convert the response column as follows:

train[, y] <- as.factor(train[, y])
test[, y] <- as.factor(test[, y]) # SalePrice 없어서 에러

#### Specify Base Learners & Metalearner
#For this example, we will use the default base learner library for `h2o.ensemble`, which includes the default H2O GLM, Random Forest, GBM and Deep Neural Net (all using default model parameter values).  We will also use the default metalearner, the H2O GLM.
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", "h2o.gbm.wrapper", "h2o.deeplearning.wrapper")
metalearner <- "h2o.deeplearning.wrapper"
family <- "AUTO"
cvControl <- list(V = 5)

#### Train an Ensemble
#Train the ensemble (using 5-fold internal CV) to generate the level-one data.  Note that more CV folds will take longer to train, but should increase performance.

slackr_bot("h2o Ensemble learning start", username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

fit <- h2o.ensemble(x = x, y = y,
                    training_frame = train,
                    family = family,
                    learner = learner,
                    metalearner = metalearner,
                    cvControl = cvControl)


## test------------------------
|model <- h2o.gbm(x = x.indep, y = y.dep, training_frame = train.h2o, distribution = "bernoulli")

perf <- h2o.performance(model, train.h2o)

h2o.auc(perd_train)



## 3_data prediction ----------------------------------------
library(mice)
library(randomForest)

rf_model <- randomForest(SalePrice ~ ., data = train, ntrees = 800,
                          nfolds = nfolds, fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE, seed = 1122)

rf_model <- randomForest(SalePrice ~ .,
                         data = train,
                         ntree = 510, mtry = 22, na.action = na.omit)

print(rf_model)

preds <- predict(rf_model, newdata = test)

results <- data.frame(Id = full_data$Id[1461:2919], SalePrice = preds)

write.csv(results, 'C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/sub_20170927_0305_rf.csv', row.names = FALSE)

varImpPlot(rf_model)

ntree <- seq(500, 2000, by = 10)
mtry <- c(10:160)
param <- data.frame(n = ntree, m = mtry)
param

for (i in param$n) {
    cat('ntree=', i, '\n')
    for (j in param$m) {
        cat('mtry', j, '\n')
        rf_model_i <- randomForest(SalePrice ~ .,
                         data = train,
                         ntree = i, mtry = j, na.action = na.omit)

        print(rf_model_i)

        ## Variance explained
        var_explained <- (100 * (1 - sum((rf_model_i$y - rf_model_i$pred) ^ 2) / sum((rf_model_i$y - mean(rf_model_i$y)) ^ 2)))

        result_1 <- c(i, j, var_explained)
        result <- rbind(result, result_1)
        colnames(result) <- c("ntree", "mtry", "var_explained")
    }
}

library("slackr")
slackr_bot(result, username = "slackr", channel = "#r_bot", incoming_webhook_url = "https://hooks.slack.com/services/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")

## etc -----------------

h2o.glm.nn <- function(..., non_negative = T, lambda_search = T) h2o.glm.wrapper(..., non_negative = non_negative, lambda_search = lambda_search)

h2o.glm.1 <- function(..., alpha = 0.5, solver = 'L_BFGS') h2o.glm.wrapper(..., alpha = alpha, solver = solver)

h2o.gbm.1 <- function(..., ntrees = 1000, stopping_rounds = 3, max_depth = 3, score_each_iteration = F) h2o.gbm.wrapper(..., ntrees = ntrees, stopping_rounds = stopping_rounds, max_depth = max_depth, score_each_iteration = score_each_iteration)
