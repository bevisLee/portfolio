# �Ҹ� ȸ�� "ABC Private Limited"�� ���� ī�װ����� �پ��� ��ǰ�� ���� 
# ���� ���� �ൿ (��ü������ ���� �ݾ�)�� �����ϰ����մϴ�. 

# �׵��� ���� ���� ���� �� �뷮 ��ǰ�� ���� �پ��� ������ ���� ����� �����߽��ϴ�.

# ������ ��Ʈ���� ���� ���� ���� �α� ���(����, ����, ��ȥ ����, ���� _ ����, ���� _ ���� _city), 
# ��ǰ ���� ����(product_id �� ��ǰ ī�װ���) �� �� ���� _ �ݾ׵� ���Ե˴ϴ�.

# ���� �׵��� �پ��� ��ǰ�� ���� ������ ���� �ݾ��� �����ϴ� ���� �����Ͽ� 
# ���� �ٸ� ��ǰ�� ���� ���������� ������ ������ ����� �� �����̵˴ϴ�.

# data definition
# User_ID                    : User ID
# Product_ID                 : Product ID
# Gender                     : Sex of User
# Age                        : Age in bins
# Occupation                 : Occupation(Masked) ����
# City_Category              : Category of the City(A, B, C)
# Stay_In_Current_City_Years : Number of years stay in current city
# Marital_Status             : Marital Status ��ȥ ����
# Product_Category_1         : Product Category(Masked)
# Product_Category_2         : Product may belongs to other category also(Masked)
# Product_Category_3         : Product may belongs to other category also(Masked)
# Purchase Purchase          : Amount(Target Variable)

if (!require("data.table")) { install.packages("data.table") }

####_1. Data load and first Prediction export -----------------------
# load data using fread
train <- fread("C:/Users/bevis/Downloads/Black_Friday/train.csv", stringsAsFactors = T)
test1 <- fread("C:/Users/bevis/Downloads/Black_Friday/test.csv", stringsAsFactors = T)
test <- test1[,-12]

#No. of rows and columns in Train, test
dim(train)
dim(test)

#Check NA data
colSums(sapply(train, is.na)) # Product_Category_2 : 173,638 / Product_Category_3 : 383,247 

# �� 12�� ���� �� 2���� ���� NA ����
# ���Ŵ� ���� ����, �������� ���� ����
# ���� ����(������)�� ������ ���캸��, �̰��� ȸ�� ������� �߷��� �� ����
# first prediction �� train data�� ��� ���������� ���̽����� ����

#first prediction using mean
sub_mean <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = mean(train$Purchase))
write.csv(sub_mean, file = "C:/Users/bevis/Downloads/Black_Friday/first_sub.csv", row.names = F)

#Check train and test
str(train) # 12 variables
str(test) # 11 variables

summary(train)
summary(test)
# Product_Category_1, 2, 3 �� �ִ밪 ����ġ / ī�װ��� ������ ���� ���� �ʿ�

#combine data set
test[, Purchase := mean(train$Purchase)]
c <- list(train, test)
combin <- rbindlist(c) # rbind ���� ����

####_2. EDA -----------------------
## ���� ���� ���� ���� : �ܺ��� �м� vs �̺��� �м�
# �ܺ���(univariate analysis) : ���Ӻ����� 1���� ��� / T-test, ANOVA, ȸ�ͺм� Ȱ��
    # T-test : �� ���� ������ ��� ���̰� �������� �м�
    # ANOVA : �� �� �̻� ������ ��� ���̰� �������� �м�
 # �̺���(bivariate analysis) : ���Ӻ��� 1��, �������� 2�� / ����м� Ȱ��
    # �� ���� �������� ���ԵǴ� ����� �м��̴�. 
    # �� �������� �������̳� ���ü��� �˾ƺ��� ���� �󵵺��� �м��̳�, 
    # �ܼ�����̳� ȸ�ͺм�, �׸��� �ϳ��� ���������� �ϳ��� ���Ӻ����� 
    # �̷������ ����� ���̰��� ���� �̺����м��� �����̴�.
 # �ٺ��� : ���Ӻ��� 2���̻� / ���κм�, �����м�, ���ػ���м�, ������ô����, �� Ȱ��

#analyzing gender variable
combin[, prop.table(table(Gender))] # Gender ���� Ȯ��
 # -> F = 1, M = 0 ���ڵ� �ʿ�

#Age Variable
combin[, prop.table(table(Age))] # Age ���� Ȯ��

#City Category Variable
combin[, prop.table(table(City_Category))] # City ���� Ȯ��

#Stay in Current Years Variable
combin[, prop.table(table(Stay_In_Current_City_Years))] # ���ֳ�� ���� Ȯ��

#unique values in ID variables
length(unique(combin$Product_ID)) # Product ID �� Ȯ��

length(unique(combin$User_ID)) # User ID �� Ȯ��

#missing values
colSums(is.na(combin))

# �ܺ��� �м��� ���� �߷�
 # Gender ������ 0,1�� ���ڵ�
 # Age�� �ٽ� �ڵ� �ؾ���
 # City_Category ���� 3���� ���ڵ�
 # Stay_In_Current_City_Years �� "4+" ������ ���� �ؾ� ��
 # 2���� ������ NA�� ����(������ ��), ������ �߼��� ������ ���� ���Ե� ��찡 ����
 # ������ ��Ʈ�� ��� ���� ID�� ���ԵǾ� ���� �ʾ�, ���� �����Ͼ �ʿ���
# ������ ���۷�(data wrangler) : �뷮�� �����͸� �Ÿ���, ��ȯ�ϰ�, ��ġ��, �з��ϴ� ������

## �ð�ȭ�� ���� ���� �ľ�
if (!require("ggplot2")) { install.packages("ggplot2") }

## ����(plot�� ���� �м�) - https://www.analyticsvidhya.com/blog/2016/03/questions-ggplot2-package-r/

# ���� �׷��� : ������ ���� �Ǵ� ���� �� ������ ������ ������ �׸� �� ���
#Age vs Gender
ggplot(combin, aes(Age, fill = Gender)) + geom_bar()

#Age vs City_Category
ggplot(combin, aes(Age, fill = City_Category)) + geom_bar()

## ������ ������ �м��ϱ� ���� ���� ���̺� ����
if (!require("gmodels")) { install.packages("gmodels") }
CrossTable(combin$Occupation, combin$City_Category)
 
####_2-1. T-test & ANOVA ------------------------
## Gender ����, T-test
t.test(Purchase ~ Gender, data = combin,
    alternative = c("two.sided"), # c("two.sided : ������", "less : ������", "greater : ū��")
    var.equal = TRUE, conf.level = 0.95)

# data:Purchase by Gender
# t = -44.802, df = 783660, p - value < 2.2e-16
# alternative hypothesis:true difference in means is not equal to 0
# 95 percent confidence interval:
# -514.7318 - 471.5829
# sample estimates:
# mean in group F mean in group M
# 8892.665 9385.823

# --> p - value < 2.2e-16 = .00000000000000022 / 2.2�� -16���

## Purchase ����, ��� ���ž� 9263.969 T-test
t.test(combin$Purchase,
    alternative = "greater",
    mu = 9263.969, con.level = 0.95)

# data:combin$Purchase
# t = -6.0381e-05, df = 783670, p - value = 0.5
# alternative hypothesis:true mean is greater than 9263.969
# 95 percent confidence interval:
# 9256.149 Inf
# sample estimates:
# mean of x
# 9263.969

## Age ����, ANOVA
bartlett.test(Purchase ~ Age, data = combin)

# Bartlett test of homogeneity of variances

# data:Purchase by Age
# Bartlett 's K-squared = 68.914, df = 6, p-value = 6.827e-13

## City_Category ����, ANOVA
bartlett.test(Purchase ~ City_Category, data = combin)

# Bartlett test of homogeneity of variances

# data:Purchase by City_Category
# Bartlett 's K-squared = 967.88, df = 2, p-value < 2.2e-16

####_3. Data wrangling use data.table ------------------------
## NA�� �����ϴ� ������ ���ο� ������ ����� ���� ������ ��
# Product_Category_2 --> Product_Category_2_NA
# Product_Category_3 --> Product_Category_3_NA

#create a new variable for missing values
combin[, Product_Category_2_NA := ifelse(sapply(combin$Product_Category_2, is.na) == TRUE, 1, 0)]
combin[, Product_Category_3_NA := ifelse(sapply(combin$Product_Category_3, is.na) == TRUE, 1, 0)]

#impute missing values : NA�� ������ ���ڷ� ��ü
combin[, Product_Category_2 := ifelse(is.na(Product_Category_2) == TRUE, "-999", Product_Category_2)]
combin[, Product_Category_3 := ifelse(is.na(Product_Category_3) == TRUE, "-999", Product_Category_3)]

#set column level

# Stay_In_Current_City_Years �� "4+" ������ ����
levels(combin$Stay_In_Current_City_Years)[levels(combin$Stay_In_Current_City_Years) == "4+"] <- "4"

#recoding age groups
levels(combin$Age)[levels(combin$Age) == "0-17"] <- 0
levels(combin$Age)[levels(combin$Age) == "18-25"] <- 1
levels(combin$Age)[levels(combin$Age) == "26-35"] <- 2
levels(combin$Age)[levels(combin$Age) == "36-45"] <- 3
levels(combin$Age)[levels(combin$Age) == "46-50"] <- 4
levels(combin$Age)[levels(combin$Age) == "51-55"] <- 5
levels(combin$Age)[levels(combin$Age) == "55+"] <- 6

## �𵨸� ������ ���� ���� ������ ���� �Ǵ� ������ ��ȯ
#convert age to numeric
combin$Age <- as.numeric(combin$Age)

#convert Gender into numeric
combin[, Gender := as.numeric(as.factor(Gender)) - 1]

#User Count : ����ں� ���ż� �ľ�
combin[, User_Count := .N, by = User_ID]

#Product Count : �α� ��ǰ �ľ�
combin[, Product_Count := .N, by = Product_ID]

#Mean Purchase of Product : ��ǰ�� ��ǰ�� ��� ���� ����(���� ������ ���߸� ������ Ȯ���� �������ų� �� �ݴ밡 �� �� ����)
combin[, Mean_Purchase_Product := mean(Purchase), by = Product_ID]

#Mean Purchase of User : ����ں��� ��� ���� ������ �����ϴ� �� �ٸ� ����, �� ����ڰ� ��� ���� �� �ݾ��� ����
combin[, Mean_Purchase_User := mean(Purchase), by = User_ID]

## City_Category 3�� ���ڵ�
if (!require("dummies")) { install.packages("dummies") }
combin <- dummy.data.frame(combin, names = c("City_Category"), sep = "_") # City_Category_A, B, C �и� 1 / 0 ���� ����

#check classes of all variables : ���� Ÿ�� Ȯ��
sapply(combin, class)

#converting Product Category 2 & 3
combin$Product_Category_2 <- as.integer(combin$Product_Category_2)
combin$Product_Category_3 <- as.integer(combin$Product_Category_3)

####_4-1. H2O use Modeling ------------------------
 # Regression: Starters Guide to Regression - https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/
 # Random Forest, GBM:Starters Guide to Tree Based Algorithms - https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
 # Deep Learning:Starters Guide to Deep Learning - https://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/

#Divide into train and test
c.train <- combin[1:nrow(train),] # 550,068
c.test <- combin[-(1:nrow(train)),]

# train data Noise ���Ÿ� ���� Product_Category_1 ~ 18������ ��� ���� �����Ͽ� 
# ī�װ��� ���� 19 & 20�� ���� �����Ͽ� ����
c.train <- c.train[c.train$Product_Category_1 <= 18,] # ���� : 550,068 --> ���� : 545,915

if (!require("h2o")) { install.packages("h2o") }

# h2o Ŭ������ ����
localH2O <- h2o.init(nthreads = -1)

# h2o �ν���Ʈ ���� Ȯ��
h2o.init()

#data to h2o cluster : �ν���Ʈ�� ������ ����
train.h2o <- as.h2o(c.train)
test.h2o <- as.h2o(c.test)

#check column index number
colnames(train.h2o)

#dependent variable (Purchase)
y.dep <- 14

#independent variables (dropping ID variables)
x.indep <- c(3:13, 15:20)

####_4-2. Multiple Regression in H2O ------------------------
regression.model <- h2o.glm(y = y.dep, x = x.indep, training_frame = train.h2o, family = "gaussian")

h2o.performance(regression.model)
# H2ORegressionMetrics:glm
# ** Reported on training data. **

# MSE:16710559
# R2:0.3261545
# Mean Residual Deviance:16710559
# Null Deviance:1.353804e+13
# Null D.o.F.:545914
# Residual Deviance:9.122545e+12
# Residual D.o.F.:545898
# AIC:10628689

#make predictions
predict.reg <- as.data.frame(h2o.predict(regression.model, test.h2o))
sub_reg <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.reg$predict)

write.csv(sub_reg, file = "C:/Users/bevis/Downloads/Black_Friday/sub_reg.csv", row.names = F)

####_4-3. Random Forest in H2O ------------------------
#Random Forest
system.time(
rforest.model <- h2o.randomForest(y = y.dep, x = x.indep, training_frame = train.h2o, ntrees = 1000, mtries = 3, max_depth = 4, seed = 1122)
)
# ����� �ý��� elapsed
# 3.70   0.25   435.67

h2o.performance(rforest.model)
# H2ORegressionMetrics:drf
# ** Reported on training data. **
# Description:Metrics reported on Out - Of - Bag training samples

# MSE:10283633
# R2:0.5853173
# Mean Residual Deviance:10283633

#check variable importance
h2o.varimp(rforest.model)

#making predictions on unseen data
system.time(predict.rforest <- as.data.frame(h2o.predict(rforest.model, test.h2o)))
# ����� �ý��� elapsed
# 1.05   0.03   31.95

#writing submission file
sub_rf <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.rforest$predict)
write.csv(sub_rf, file = "C:/Users/bevis/Downloads/Black_Friday/sub_rf.csv", row.names = F)

####_4-4. GBM in H2O ------------------------
#GBM
system.time(
gbm.model <- h2o.gbm(y = y.dep, ## target: using the logged variable created earlier
                     x = x.indep, ## this can be names or column numbers
                     training_frame = train.h2o, ## H2O frame holding the training data
                     ntrees = 1000, ## use fewer trees than default (50) to speed up training
                     max_depth = 4, 
                     learn_rate = 0.01, ## lower learn_rate is better, but use high rate to offset few trees
                     seed = 1122)
)
# ����� �ý��� elapsed
# 8.47   0893   935.09

h2o.performance(gbm.model)
# H2ORegressionMetrics:gbm
# ** Reported on training data. **

# MSE:6319672
# R2:0.7451622
# Mean Residual Deviance:6319672

#making prediction and writing submission file
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.gbm$predict)
write.csv(sub_gbm, file = "C:/Users/bevis/Downloads/Black_Friday/sub_gbm.csv", row.names = F)

####_4-5. Deep-Learning in H2O ------------------------
#deep learning models
system.time(
             dlearning.model <- h2o.deeplearning(y = y.dep,
             x = x.indep,
             training_frame = train.h2o,
             epoch = 60,
             hidden = c(100, 100),
             activation = "Rectifier",
             seed = 1122
             )
)
# ����� �ý��� elapsed
# 5.50   0.36   595.75

h2o.performance(dlearning.model)
# H2ORegressionMetrics:deeplearning
# ** Reported on training data. **
# Description:Metrics reported on temporary training frame with 10017 samples

# MSE:5782595
# R2:0.7611859
# Mean Residual Deviance:5782595

#making predictions
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))

#create a data frame and writing submission file
sub_dlearning <- data.frame(User_ID = test$User_ID, Product_ID = test$Product_ID, Purchase = predict.dl2$predict)
write.csv(sub_dlearning, file = "C:/Users/bevis/Downloads/Black_Friday/sub_dlearning_new.csv", row.names = F)

####_5. Next To Do ------------------------
# GBM, Deep Learning �� Random Forest���� �Ű� ���� Ʃ���� �����Ͻʽÿ�.
# �Ű� ���� Ʃ�׿� �׸��� �˻��� ����Ͻʽÿ�. H2O������ �۾��� �����ϴ� h2o.grid��� ���� �Լ��� �ֽ��ϴ�.
# �𵨿� ���ο� ������ ������ ���ִ� �� ���� ����� ����� ����� �����غ��ʽÿ�.
# ����������, �� ���� ���� ��� ���� ��� ����� �ӻ�� ó���Ͻʽÿ�.
