
## ÂüÁ¶ - http://rstudio-pubs-static.s3.amazonaws.com/210076_4d3084347a924ae6b09d753ed2db4749.html

# Loading packages and setting the random seed
if (!require("caret")) { install.packages("caret") }
if (!require("knitr")) { install.packages("knitr") }
if (!require("MASS")) { install.packages("MASS") }
if (!require("dplyr")) { install.packages("dplyr") }
if (!require("mice")) { install.packages("mice") }
if (!require("xgboost")) { install.packages("xgboost") }

set.seed(1234)

# Loading the training and testing data sets
train = read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train.csv")
test = read.csv("C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test.csv")

training <- train
testing <- test

# Data Cleaning and transformations
missing.summary <- sapply(training, function(x) sum(is.na(x)))

indexs.missing <- sapply(training, function(x) sum(is.na(x))) > 0

num.variable.missing <- length(missing.summary[indexs.missing])

freq.table.miss <- data.frame(Variable = names(missing.summary[indexs.missing]), Number.of.Missing = as.integer(missing.summary[indexs.missing]), Porcentage.of.Missing = as.numeric(prop.table(missing.summary[indexs.missing])))

freq.table.miss <- freq.table.miss %>% select(Variable:Porcentage.of.Missing) %>% arrange(desc(Number.of.Missing))

# We exclude the variables with many missing values

indexs <- missing.summary < 690

training <- training[, indexs]

# We retain SalePrice 

SalePrice <- training$SalePrice

# We exclude the first variable and Sale Price 

training <- training %>%
            select(-Id,- SalePrice)

indexs.quantitative <- sapply(training, function(x) is.numeric(x))

# We split the train data set into quantitative variables and cualitatives variables.

training.quantitative <- training[, indexs.quantitative]

training.qualitative <- training[, !indexs.quantitative]

nzv <- nzv(training.qualitative, saveMetrics = TRUE)

kable(nzv)

training.qualitative <- training.qualitative[, !nzv$nzv]

nzv2 <- nzv(training.quantitative, saveMetrics = TRUE)

kable(nzv)

training.quantitative <- training.quantitative[, !nzv2$nzv]

tempData <- mice(training.quantitative, m = 5, maxit = 50, meth = 'pmm', seed = 1234, printFlag = FALSE)

training.quantitative.imputed <- complete(tempData, 1)

pre.proc <- preProcess(training.quantitative.imputed, method = c("center", "scale", "pca"), thresh = 0.90)

training.quantitative.imputed.pc <- predict(pre.proc, training.quantitative.imputed)

## Fitting and running XGBoost model
dummies <- dummyVars(~., data = training.qualitative)
training.dummies <- as.data.frame(predict(dummies, training.qualitative))

training.imputed <- cbind(training.dummies, training.quantitative.imputed.pc)

training.imputed$SalePrice <- SalePrice

# We split the data set into 2 parts: training data (90%) and testing data (20%).
inTrain <- createDataPartition(y = training.imputed$SalePrice,
                               p = 0.90,
                               list = FALSE)

num.variables <- dim(training.imputed)[2]

train.xgboost <- training.imputed[inTrain,]

test.xgboost <- training.imputed[-inTrain,]

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

## Predicting with the XGBoost model
n.indexs <- length(indexs)
indexs <- indexs[-n.indexs]

testing <- testing[, indexs]

# We retain Id 

Id <- testing$Id

# We exclude the first variable

testing <- testing %>%
            select(-Id)

indexs.quantitative <- sapply(testing, function(x) is.numeric(x))

# We split the train data set into quantitative variables and cualitatives variables.

testing.quantitative <- testing[, indexs.quantitative]

testing.qualitative <- testing[, !indexs.quantitative]

testing.qualitative <- testing.qualitative[, !nzv$nzv]

testing.quantitative <- testing.quantitative[, !nzv2$nzv]

tempData <- mice(testing.quantitative, m = 5, maxit = 50, meth = 'pmm', seed = 1234, printFlag = FALSE)

testing.quantitative.imputed <- complete(tempData, 1)

testing.quantitative.imputed.pc <- predict(pre.proc, testing.quantitative.imputed)

testing.dummies <- as.data.frame(predict(dummies, testing.qualitative))

testing.imputed <- cbind(testing.dummies, testing.quantitative.imputed.pc)

pred.testing.xgboost <- predict(house.xgboost, data.matrix(testing.imputed), missing = NA)

submission <- data.frame(Id = Id, SalePrice = pred.testing.xgboost)

write.csv(submission, file = "C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/xgboost_sub.csv", row.names = FALSE, quote = FALSE)