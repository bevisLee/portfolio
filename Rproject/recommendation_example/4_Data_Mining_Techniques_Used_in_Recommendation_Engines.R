####--------------------------------------- 
# Data Mining Techniques Used in Recommendation Engines
####--------------------------------------- 

####--------------------------------------- 
## 1. Neighbourhood-based techniques
  ## Euclidean distance
x1 <- rnorm(30)
x2 <- rnorm(30)
Euc_dist = dist(rbind(x1,x2) ,method="euclidean")

  ## Cosine similarity
vec1 = c( 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
vec2 = c( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0 )

if(!require("lsa")) { install.packages("lsa")}
cosine(vec1,vec2)

  ## Jaccard similarity
vec1 = c( 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
vec2 = c( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0 )

if(!require("clusteval")) { install.packages("clusteval")}
cluster_similarity(vec1, vec2, similarity = "jaccard")

  ## Pearson correlation coefficient
Coef = cor(mtcars, method="pearson")

####--------------------------------------- 
## 2. Mathematical modelling techniques
  ## Matrix factorization
#MF
if(!require("recommenderlab")) { install.packages("recommenderlab")}
data("MovieLense")
dim(MovieLense)

#applying MF using NMF
mat = as(MovieLense,"matrix")
mat[is.na(mat)] = 0

if(!require("NMF")) { install.packages("NMF")}
res = nmf(mat,10)
res

#fitted values
r.hat <- fitted(res)
dim(r.hat)
p <- basis(res)
dim(p)
q <- coef(res)
dim(q)

  ## Alternating Least Squares

  ## Singular value decomposition
# 참고 url - http://rfriend.tistory.com/185
sampleMat <- function(n) { i <- 1:n; 1 / outer(i - 1, i, "+") }
original.mat <- sampleMat(9)[, 1:6]

if(!require("svd")) { install.packages("svd")}
(s <- svd(original.mat))
D <- diag(s$d)

# X = U D V'
s$u %*% D %*% t(s$v)

####--------------------------------------- 
## 3. Machine learning techniques
  ## Linear regression
if(!require("MASS")) { install.packages("MASS")}
data("Boston")
set.seed(0)
which_train <- sample(x = c(TRUE, FALSE), size = nrow(Boston),
                      replace = TRUE, prob = c(0.8, 0.2))
train <- Boston[which_train, ]
test <- Boston[!which_train, ]
lm.fit =lm(medv~. ,data=train )
summary(lm.fit)

#predict new values
pred = predict(lm.fit,test[,-14])

  ## Classification models
set.seed(1)
x1 = rnorm(1000) # sample continuous variables
x2 = rnorm(1000)
z = 1 + 4*x1 + 3*x2 # data creation
pr = 1/(1+exp(-z)) # applying logit function
y = rbinom(1000,1,pr) # bernoulli response variable

#now feed it to glm:
df = data.frame(y=y,x1=x1,x2=x2)
glm(y~x1+x2,data=df,family="binomial")

  ## KNN classification
data("iris")

if(!require("dplyr")) { install.packages("dplyr")}
iris2 = sample_n(iris, 150)
train = iris2[1:120,]
test = iris2[121:150,]
cl = train$Species

if(!require("caret")) { install.packages("caret")}
fit <- knn3(Species~., data=train, k=3)
predictions <- predict(fit, test[,-5], type="class")
table(predictions, test$Species)

  ## Support vector machines
if(!require("e1071")) { install.packages("e1071")}
data(iris)
sample = iris[sample(nrow(iris)),]
train = sample[1:105,]
test = sample[106:150,]
tune =tune(svm,Species~.,data=train,kernel ="radial",scale=FALSE,ranges
           =list(cost=c(0.001,0.01,0.1,1,5,10,100)))
tune$best.model

summary(tune)

model =svm(Species~.,data=train,kernel ="radial",cost=10,scale=FALSE)

pred = predict(model,test)

  ## Decision trees
if(!require("tree")) { install.packages("tree")}
data(iris)
sample = iris[sample(nrow(iris)),]
train = sample[1:105,]
test = sample[106:150,]
model = tree(Species~.,train)
summary(model)

plot(model) #plot trees
text(model) #apply text

pred = predict(model,test[,-5],type="class")

  ## Ensemble methods
# Randomforests
if(!require("randomForest")) { install.packages("randomForest")}
data(iris)
sample = iris[sample(nrow(iris)),]
train = sample[1:105,]
test = sample[106:150,]
model =randomForest(Species~.,data=train,mtry=2,importance =TRUE,proximity=TRUE)

pred = predict(model,newdata=test[,-5])

  ## Bagging
# Boosting
if(!require("gbm")) { install.packages("gbm")}

data(iris)
sample = iris[sample(nrow(iris)),]
train = sample[1:105,]
test = sample[106:150,]
model = gbm(Species~.,data=train,distribution="multinomial",n.trees=5000,interaction.depth=4)
summary(model)

pred = predict(model,newdata=test[,-5],n.trees=5000)
p.pred <- apply(pred,1,which.max)

####--------------------------------------- 
## 4. Clustering techniques
  # K-means clustering
if(!require("cluster")) { install.packages("cluster")}
data(iris)
iris$Species = as.numeric(iris$Species)
kmeans<- kmeans(x=iris, centers=5)
clusplot(iris,kmeans$cluster, color=TRUE, shade=TRUE,labels=13, lines=0)

if(!require("ggplot2")) { install.packages("ggplot2")}
data(iris)
iris$Species = as.numeric(iris$Species)
cost_df <- data.frame()

for(i in 1:100){
  kmeans<- kmeans(x=iris, centers=i, iter.max=50)
  cost_df<- rbind(cost_df, cbind(i, kmeans$tot.withinss))
}

names(cost_df) <- c("cluster", "cost")

#Elbow method to identify the idle number of Cluster

#Cost plot
ggplot(data=cost_df, aes(x=cluster, y=cost, group=1)) +
  theme_bw(base_family="Garamond") +
  geom_line(colour = "darkgreen") +
  theme(text = element_text(size=20)) +
  ggtitle("Reduction In Cost For Values of 'k'\n") +xlab("\nClusters") + ylab("Within-Cluster Sum of Squares\n")

####--------------------------------------- 
## 5. Dimensionality reduction
  # Principal component analysis
data(USArrests)
head(states)

names(USArrests)

apply(USArrests , 2, var)

pca =prcomp(USArrests , scale =TRUE)

names(pca)

pca$rotation=-pca$rotation
pca$x=-pca$x
biplot (pca , scale =0)

####--------------------------------------- 
## 6. Vector space models
# 참고 url - https://rstudio-pubs-static.s3.amazonaws.com/132792_864e3813b0ec47cb95c7e1e2e2ad83e7.html
  ## Term frequency

  ## Term frequency-inverse document frequency
if(!require("tm")) { install.packages("tm")}
data(crude)

tdm <- TermDocumentMatrix(crude,control=list(weighting = function(x) weightTfIdf(x, normalize =TRUE), stopwords = TRUE))
inspect(tdm)

####--------------------------------------- 
## 7. Evaluation techniques
  # Root-mean-square error
  # Mean absolute error
  # Precision and recall

