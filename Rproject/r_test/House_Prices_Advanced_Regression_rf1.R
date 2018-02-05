## kernal - https://www.kaggle.com/captcalculator/random-forest-missing-data-imputed-lb-0-15604

library(mice)
library(randomForest)

# Get the data
train_raw <- read.csv('C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/train.csv', stringsAsFactors = FALSE)
test_raw <- read.csv('C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/test.csv', stringsAsFactors = FALSE)

# Convert MSSubClass, MoSold, and YrSold to factors
train_raw$MSSubClass <- as.factor(train_raw$MSSubClass)
test_raw$MSSubClass <- as.factor(test_raw$MSSubClass)

train_raw$MoSold <- as.factor(train_raw$MoSold)
test_raw$MoSold <- as.factor(test_raw$MoSold)

train_raw$YrSold <- as.factor(train_raw$YrSold)
test_raw$YrSold <- as.factor(test_raw$YrSold)


## Fill in missing data. 
## Replace NAs with 'None' where component is missing. 
## Use mice to impute remaining missing data

# Replace NAs with 'None' for no pool
train_raw$PoolQC[is.na(train_raw$PoolQC)] <- 'None'
test_raw$PoolQC[is.na(test_raw$PoolQC)] <- 'None'

# Replace NAs with 'None' for no alley
train_raw$Alley[is.na(train_raw$Alley)] <- 'None'
test_raw$Alley[is.na(test_raw$Alley)] <- 'None'

# Replace NAs with 'None' for no misc feature
train_raw$MiscFeature[is.na(train_raw$MiscFeature)] <- 'None'
test_raw$MiscFeature[is.na(test_raw$MiscFeature)] <- 'None'

# Replace NAs with 'None' for no fence
train_raw$Fence[is.na(train_raw$Fence)] <- 'None'
test_raw$Fence[is.na(test_raw$Fence)] <- 'None'

# Replace NAs with 'None' for no fireplace
train_raw$FireplaceQu[is.na(train_raw$FireplaceQu)] <- 'None'
test_raw$FireplaceQu[is.na(test_raw$FireplaceQu)] <- 'None'

# Replace NAs with 'None' for no garage
train_raw$GarageType[is.na(train_raw$GarageType)] <- 'None'
test_raw$GarageType[is.na(test_raw$GarageType)] <- 'None'

# Replace NAs with 'None' for no garage
train_raw$GarageFinish[is.na(train_raw$GarageFinish)] <- 'None'
test_raw$GarageFinish[is.na(test_raw$GarageFinish)] <- 'None'

# Replace NAs with 'None' for no garage
train_raw$GarageQual[is.na(train_raw$GarageQual)] <- 'None'
test_raw$GarageQual[is.na(test_raw$GarageQual)] <- 'None'

# Replace NAs with 'None' for no garage
train_raw$GarageCond[is.na(train_raw$GarageCond)] <- 'None'
test_raw$GarageCond[is.na(test_raw$GarageCond)] <- 'None'

# Replace NAs with 'None' for no basement
train_raw$BsmtExposure[is.na(train_raw$BsmtExposure)] <- 'None'
test_raw$BsmtExposure[is.na(test_raw$BsmtExposure)] <- 'None'

# Replace NAs with 'None' for no basement
train_raw$BsmtFinType1[is.na(train_raw$BsmtFinType1)] <- 'None'
test_raw$BsmtFinType1[is.na(test_raw$BsmtFinType1)] <- 'None'

# Replace NAs with 'None' for no basement
train_raw$BsmtFinType2[is.na(train_raw$BsmtFinType2)] <- 'None'
test_raw$BsmtFinType2[is.na(test_raw$BsmtFinType2)] <- 'None'

# Replace NAs with 'None' for no basement
train_raw$BsmtQual[is.na(train_raw$BsmtQual)] <- 'None'
test_raw$BsmtQual[is.na(test_raw$BsmtQual)] <- 'None'

# Replace NAs with 'None' for no basement
train_raw$BsmtCond[is.na(train_raw$BsmtCond)] <- 'None'
test_raw$BsmtCond[is.na(test_raw$BsmtCond)] <- 'None'

# Replace NAs with 0
train_raw$BsmtFinSF1[is.na(train_raw$BsmtFinSF1)] <- 0
test_raw$BsmtFinSF1[is.na(test_raw$BsmtFinSF1)] <- 0

# Replace NAs with 0
train_raw$BsmtFinSF2[is.na(train_raw$BsmtFinSF2)] <- 0
test_raw$BsmtFinSF2[is.na(test_raw$BsmtFinSF2)] <- 0

# Replace NAs with 0
train_raw$BsmtUnfSF[is.na(train_raw$BsmtUnfSF)] <- 0
test_raw$BsmtUnfSF[is.na(test_raw$BsmtUnfSF)] <- 0

# Replace NAs with 0
train_raw$TotalBsmtSF[is.na(train_raw$TotalBsmtSF)] <- 0
test_raw$TotalBsmtSF[is.na(test_raw$TotalBsmtSF)] <- 0

# Replace NAs with 'None' for no mason veneer
train_raw$MasVnrType[is.na(train_raw$MasVnrType)] <- 'None'
test_raw$MasVnrType[is.na(test_raw$MasVnrType)] <- 'None'

# Replace NAs with 0
train_raw$MasVnrArea[is.na(train_raw$MasVnrArea)] <- 0
test_raw$MasVnrArea[is.na(test_raw$MasVnrArea)] <- 0

# Function to convert character columns back to factor
toFactor <- function(x) {
    if (is.character(x)) {
        factor(x)
    }
    return(x)
}

# Apply toFactor to train and test data
train_raw <- as.data.frame(lapply(train_raw, toFactor))
test_raw <- as.data.frame(lapply(test_raw, toFactor))

# Impute the remaining missing values with MICE
imp.train_raw <- mice(train_raw, m = 3, method = 'cart', maxit = 1)
imp.test_raw <- mice(test_raw, m = 3, method = 'cart', maxit = 1)

# Merge the imputed values with the original data
train <- complete(imp.train_raw)
test <- complete(imp.test_raw)

# MICE had trouble with this one. Replace NA with most common type
test$Utilities[is.na(test$Utilities)] <- 'AllPub'

## Preprocess

# Add factor level 150 to MSSubClass in train. (150 is in test but not in train)
levels(train$MSSubClass) <- c(levels(train$MSSubClass), 150)

levels(test$Utilities) <- levels(train$Utilities)
levels(test$Condition2) <- levels(train$Condition2)
levels(test$HouseStyle) <- levels(train$HouseStyle)
levels(test$RoofMatl) <- levels(train$RoofMatl)
levels(test$Exterior1st) <- levels(train$Exterior1st)
levels(test$Exterior2nd) <- levels(train$Exterior2nd)
levels(test$Heating) <- levels(train$Heating)
levels(test$Electrical) <- levels(train$Electrical)
levels(test$GarageQual) <- levels(train$GarageQual)
levels(test$PoolQC) <- levels(train$PoolQC)
levels(test$MiscFeature) <- levels(train$MiscFeature)


## Random forest model

rf_model <- randomForest(SalePrice ~ . - Id,
                         data = train,
                         ntree = 800)

print(rf_model)

preds <- predict(rf_model, newdata = test)

results <- data.frame(Id = test$Id, SalePrice = preds)

write.csv(results, 'C:/Users/bevis/Downloads/House_Prices_Advanced_Regression_Techniques/sub_mission_rf.csv', row.names = FALSE)


