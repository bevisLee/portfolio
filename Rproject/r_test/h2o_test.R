##=========================================================================
## 01. H2O ¼³Ä¡: 
##=========================================================================

# The following two commands remove any previously installed H2O packages for R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload = TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# Next, we download packages that H2O depends on.
if (!("methods" %in% rownames(installed.packages()))) { install.packages("methods") }
if (!("statmod" %in% rownames(installed.packages()))) { install.packages("statmod") }
if (!("stats" %in% rownames(installed.packages()))) { install.packages("stats") }
if (!("graphics" %in% rownames(installed.packages()))) { install.packages("graphics") }
if (!("RCurl" %in% rownames(installed.packages()))) { install.packages("RCurl") }
if (!("rjson" %in% rownames(installed.packages()))) { install.packages("rjson") }
if (!("tools" %in% rownames(installed.packages()))) { install.packages("tools") }
if (!("utils" %in% rownames(installed.packages()))) { install.packages("utils") }

# Now we download, install and initialize the H2O package for R.
install.packages("h2o", type = "source", repos = (c("http://h2o-release.s3.amazonaws.com/h2o/rel-simons/7/R")))
library(h2o)

localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '4g')

h2o.clusterInfo()

h2o.shutdown()

##=========================================================================
## 02. Data import
##=========================================================================
h2o.init(nthreads = -1)

h2o.no_progress() # Disable progress bars for Rmd

# This step takes a few seconds bc we have to download the data from the internet...
train_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz"
test_file <- "https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz"
train <- h2o.importFile(train_file)
test <- h2o.importFile(test_file)

y <- "C785" #response column: digits 0-9
x <- setdiff(names(train), y) #vector of predictor column names

# Since the response is encoded as integers, we need to tell H2O that
# the response is in fact a categorical/factor column.  Otherwise, it 
# will train a regression model instead of multiclass classification.
train[, y] <- as.factor(train[, y])
test[, y] <- as.factor(test[, y])