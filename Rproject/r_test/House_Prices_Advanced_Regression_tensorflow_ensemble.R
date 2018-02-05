## EDA data import-----------------------
x.train <- read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train_eda.csv")

x.test <- read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test_eda.csv")

x.train <- x.train[, -1]
x.test <- x.test[, -1]

SalePrice <- x.train[, 82]


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


## model predction-----------------------
### xgboost : LB 0.12292
if (!require("xgboost")) { install.packages("xgboost") }
if (!require("Metrics")) { install.packages("Metrics") }

Dtrain <- xgb.DMatrix(as.matrix(X_train), label = SalePrice)
dtest <- xgb.DMatrix(as.matrix(X_test))

model_xgb <- xgboost(data = Dtrain, nfold = 5, showsd = TRUE,
               metrics = "rmse", verbose = FALSE, "eval_metric" = "rmse",
               "objective" = "reg:linear", "max.depth" = 6, "eta" = 0.01,
               "subsample" = 0.2,
               "colsample_bytree" = 0.2,
               nrounds = 2200, gamma = 0.0,
               min_child_weight = 1.5,
               nthread = 8, booster = "gbtree")

pred_train_xgb <- log(predict(model_xgb, Dtrain))
pred_test_xgb <- log(predict(model_xgb, dtest))

### gbm : LB 0.12600
if (!require("iterators")) { install.packages("iterators") }
if (!require("parallel")) { install.packages("parallel") }

CARET.TRAIN.CTRL <- trainControl(method = "repeatedcv", number = 5, repeats = 5, verboseIter = FALSE, allowParallel = TRUE)

gbmfit <- train(SalePrice ~ ., method = "gbm", metric = "RMSE", maximize = FALSE,
             trControl = CARET.TRAIN.CTRL,
             tuneGrid = expand.grid(n.trees = c(2000), interaction.depth = c(5), shrinkage = c(0.005), n.minobsinnode = c(15)),
                data = train_XY, verbose = FALSE)

pred_train_gbm <- log(predict(gbmfit, train_XY))
pred_test_gbm <- log(predict(gbmfit, X_test))

### randomForest : LB 0.14595
if (!require("randomForest")) { install.packages("randomForest") }
if (!require("caret")) { install.packages("caret") }

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid")

tunegrid <- expand.grid(.mtry = 31)
ntree = 1000

rf_model <- train(SalePrice ~ ., data = train_XY, method = "rf",
                 tuneGrid = tunegrid, trControl = control, ntree = ntree)

pred_train_rf <- log(predict(rf_model, train_XY))
pred_test_rf <- log(predict(rf_model, X_test))

### lasso : 0.15485
if (!require("glmnet")) { install.packages("glmnet") }
if (!require("Metrics")) { install.packages("Metrics") }

set.seed(123)
cv_lasso = cv.glmnet(as.matrix(X_train), SalePrice, alpha = 0.00012, nfolds = 10, type.measure = "mse")

pred_train_lasso <- log(predict(cv_lasso, newx = as.matrix(X_train),
    s = "lambda.min"))
pred_test_lasso <- log(predict(cv_lasso, newx = as.matrix(X_test),
    s = "lambda.min"))


## tensorflow ensemble : LB 0.12289 -----------------------
# python 참고 - https://www.kaggle.com/einstalek/blending-with-tensorflow-0-11417-on-lb/notebook
# R-tensorflow 참고 - http://cinema4dr12.tistory.com/1155

if (!require("reticulate")) { install.packages("reticulate") }
if (!require("tensorflow")) { install.packages("tensorflow") }

np <- import("numpy")

# Training Data
pred_train <- data.frame(pred_train_xgb, pred_train_gbm)
    # data.frame(pred_train_xgb, pred_train_lasso, pred_train_gbm, pred_train_rf)
preds_train = np$array(as.matrix(pred_train))

SalePrice <- log(SalePrice)
actual_price = np$array(as.matrix(data.frame(SalePrice)))

n_x = nrow(preds_train)
m = ncol(preds_train)

P <- tf$placeholder(tf$float32, name = "Preds", shape = list(n_x, m))
Y <- tf$placeholder(tf$float32, name = "Price", shape = list(n_x, 1))
A <- tf$get_variable(name = "Params", dtype = tf$float32,
                         initializer = tf$constant(np$array(matrix(c(1:m), ncol = 1, nrow = m)), dtype = tf$float32))

# loss
prediction <- tf$matmul(P, A) / tf$reduce_sum(A)
lmbda = 0.8
loss = tf$reduce_mean(tf$squared_difference(prediction, Y)) + lmbda * tf$reduce_mean(tf$abs(A))

# optimizer
optimizer = tf$train$GradientDescentOptimizer(learning_rate = 0.01)$minimize(loss)

init = tf$global_variables_initializer()

# Launch the default graph.
sess <- tf$Session()

# Fit all training data
num_iterations = 700

costs1 <- data.frame()
## tensorflow loss tuning
with(tf$Session() %as% sess, {
    # initialize global variables
    for (i in (0:num_iterations)) {
        sess$run(init)
            current_cost_optimizer = sess$run(optimizer, feed_dict = dict(P = preds_train, Y = actual_price))
            current_cost_loss = sess$run(loss, feed_dict = dict(P = preds_train, Y = actual_price))
            costs1 <- current_cost_loss
            print(i)
            print(current_cost_loss)
    }
    parameters = sess$run(A)
    print(parameters)
})

params <- np$array(as.matrix(parameters))

pred_test <- data.frame(pred_test_xgb, pred_test_gbm)
    # data.frame(pred_test_xgb, pred_test_lasso, pred_test_gbm, pred_test_rf)
preds_test = np$array(as.matrix(pred_test))

op <- data.frame()
for (i in 1:nrow(pred_test)) {
    a = sum(pred_test[i,] * params)
    op <- rbind(op, a[1])
}

op = np$array(as.matrix(op))

WAP = np$squeeze(op / np$sum(params))

WAP <- exp(WAP)

test_id <- c(1461:2919)

ensemble_pred <- data.frame(test_id, WAP)
colnames(ensemble_pred) <- c("Id", "SalePrice")

write.csv(ensemble_pred, paste0("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/tensorflow_ensemble_2_", format(Sys.time(), "%Y%b%d_%H%M%S"), ".csv ", sep = " "), quote = F, row.names = F)

head(ensemble_pred)

