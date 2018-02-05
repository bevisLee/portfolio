# run this to install keras and tensorflow
# devtools::install_github("rstudio/keras")
# install_tensorflow()

library(tidyverse)
library(kerasR)
library(tensorflow)

### Data Preparation

set.seed(1234)

train <- read_csv("C:/Users/bevis/Downloads/Mercedes-Benz_Greener_Manufacturing/train.csv")
test <- read_csv("C:/Users/bevis/Downloads/Mercedes-Benz_Greener_Manufacturing/test.csv")

train_id <- train$ID
train_labels <- train$y
test_id <- test$ID

train$y <- NULL
train$ID <- NULL
test$ID <- NULL

ntrain <- nrow(train)

# one hot encoding
df_all <- rbind(train, test)
ohe_feats = paste0("X", c(0, 1, 2, 3, 4, 5, 6, 8))
for (f in ohe_feats) {
    feat_names <- paste0(f, "_", unique(df_all[[f]]))
    df_all <- cbind(df_all, do.call(rbind, lapply(df_all[[f]], function(x) {
        y <- rep(0, length(feat_names));
        y[x] <- 1;
        names(y) <- feat_names
        return(y)
    })))

    df_all[[f]] <- NULL
}

train <- df_all[1:ntrain,]
test <- df_all[(ntrain + 1):nrow(df_all),]

### Setup Network Topology

model <- keras_model_sequential()
model %>%
  layer_dense(units = ncol(train), input_shape = c(ncol(train)), activation = "relu", kernel_initializer = 'normal') %>%
  layer_dense(units = 36, activation = "relu", kernel_initializer = 'normal') %>%
  layer_dense(units = 6, activation = "relu", kernel_initializer = 'normal') %>%
  layer_dense(units = 1, activation = "relu", kernel_initializer = 'normal')

summary(model)

### Evaluate Model

model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'mse')

train.ind <- sample(1:nrow(train), 0.8 * nrow(train))

model %>% fit(as.matrix(train[train.ind,]), as.matrix(train_labels[train.ind]), epochs = 100, batch_size = 128)

score <- model %>% evaluate(as.matrix(train[-train.ind,]), train_labels[-train.ind], batch_size = 128)

sqrt(score) #rmse

### Make Submission

model %>% fit(as.matrix(train), as.matrix(train_labels), epochs = 100, batch_size = 128)

pred <- data.frame(ID = test_id, y = predict(model, as.matrix(test)))
write.csv(pred, "submission.csv", quote = F, row.names = F)