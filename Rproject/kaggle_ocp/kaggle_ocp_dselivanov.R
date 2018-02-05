

### 1. conf.R ------------------------------
UUID_HASH_SIZE = 2 ** 30
events_h_size = 2 ** 24
views_h_size = 2 ** 24
N_PART = 128

# physical cores
N_CORES_PREPROCSESSING = 4
# cores + hyperthreads
N_THREAD_FTRL = 8

RAW_DATA_PATH = "E:/kaggle_Outbrain_Click_Prediction"

RDS_DATA_PATH = "E:/kaggle_Outbrain_Click_Prediction/rds"
if (!dir.exists(RDS_DATA_PATH))
	dir.create(RDS_DATA_PATH)

PAGE_VIEWS_CHUNKS_PATH = "E:/kaggle_Outbrain_Click_Prediction/page_views_chunks"

# # LEAK
LEAK_PATH = sprintf("%s/leak.rds", RDS_DATA_PATH)

## MODEL SAVE PATHS 
PATH_MODEL_1 = sprintf("%s/model_1.rds", RDS_DATA_PATH)
PATH_MODEL_1_SUBMISSION_FILE = sprintf("%s/model_1_submission.csv", RAW_DATA_PATH)
PATH_MODEL_2 = sprintf("%s/model_2.rds", RDS_DATA_PATH)
PATH_MODEL_2_SUBMISSION_FILE = sprintf("%s/model_2_submission.csv", RAW_DATA_PATH)
PATH_MODEL_2_SUBMISSION_FILE_LEAK = sprintf("%s/model_2_submission_leak.csv", RAW_DATA_PATH)

# # PAGE VIEWS processing folders
VIEWS_INTERMEDIATE_DIR = sprintf("%s/views_filter/", RDS_DATA_PATH)
if (!dir.exists(VIEWS_INTERMEDIATE_DIR)) dir.create(VIEWS_INTERMEDIATE_DIR)
for (i in 0L:(N_PART - 1L)) {
	d = sprintf("%s/%03d/", VIEWS_INTERMEDIATE_DIR, i)
	if (!dir.exists(d)) dir.create(d)
	}
VIEWS_DIR = sprintf("%s/views", RDS_DATA_PATH)
if (!dir.exists(VIEWS_DIR)) dir.create(VIEWS_DIR)

## BASELINE_1 data folders
RDS_BASELINE_1_MATRIX_DIR = sprintf("%s/baseline_1/", RDS_DATA_PATH)
if (!dir.exists(RDS_BASELINE_1_MATRIX_DIR)) dir.create(RDS_BASELINE_1_MATRIX_DIR)

RDS_BASELINE_1_MATRIX_DIR_TRAIN = sprintf("%strain", RDS_BASELINE_1_MATRIX_DIR)
if (!dir.exists(RDS_BASELINE_1_MATRIX_DIR_TRAIN)) dir.create(RDS_BASELINE_1_MATRIX_DIR_TRAIN)

RDS_BASELINE_1_MATRIX_DIR_CV = sprintf("%scv", RDS_BASELINE_1_MATRIX_DIR)
if (!dir.exists(RDS_BASELINE_1_MATRIX_DIR_CV)) dir.create(RDS_BASELINE_1_MATRIX_DIR_CV)

RDS_BASELINE_1_MATRIX_DIR_TEST = sprintf("%stest", RDS_BASELINE_1_MATRIX_DIR)
if (!dir.exists(RDS_BASELINE_1_MATRIX_DIR_TEST)) dir.create(RDS_BASELINE_1_MATRIX_DIR_TEST)

## BASELINE_2 data folders
RDS_BASELINE_2_MATRIX_DIR = sprintf("%s/baseline_2/", RDS_DATA_PATH)
if (!dir.exists(RDS_BASELINE_2_MATRIX_DIR)) dir.create(RDS_BASELINE_2_MATRIX_DIR)

RDS_BASELINE_2_MATRIX_DIR_TRAIN = sprintf("%strain", RDS_BASELINE_2_MATRIX_DIR)
if (!dir.exists(RDS_BASELINE_2_MATRIX_DIR_TRAIN)) dir.create(RDS_BASELINE_2_MATRIX_DIR_TRAIN)

RDS_BASELINE_2_MATRIX_DIR_CV = sprintf("%scv", RDS_BASELINE_2_MATRIX_DIR)
if (!dir.exists(RDS_BASELINE_2_MATRIX_DIR_CV)) dir.create(RDS_BASELINE_2_MATRIX_DIR_CV)

RDS_BASELINE_2_MATRIX_DIR_TEST = sprintf("%stest", RDS_BASELINE_2_MATRIX_DIR)
if (!dir.exists(RDS_BASELINE_2_MATRIX_DIR_TEST)) dir.create(RDS_BASELINE_2_MATRIX_DIR_TEST)


### 2. misc.R ------------------------------

# source("conf.R")

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(doParallel))
library(methods)
library(Matrix)
library(magrittr)

if (.Platform$OS.type != "unix") {
	cl <- makePSOCKcluster(N_CORES_PREPROCSESSING)
	registerDoParallel(cl)
	message(sprintf("Detected Windows platform. Cluster with %d cores and name \"cl\"  registred. 
                  Stop it with `stopCluster(cl)` at the end.", N_CORES_PREPROCSESSING))
} else {
	registerDoParallel(N_CORES_PREPROCSESSING)
}

fread_zip = function(file, ...) {
	fn = basename(file)
	file = path.expand(file)
	# cut ".zip" suffix using substr
	path = paste("unzip -p", file, substr(x = fn, 1, length(fn) - 4))
	fread(path, ...)
}

string_hasher = function(x, h_size = UUID_HASH_SIZE) {
	text2vec:::hasher(x, h_size)
}

save_rds_compressed = function(x, file, compression_level = 1L) {
	con = gzfile(file, open = "wb", compression = compression_level)
	saveRDS(x, file = con)
	close.connection(con)
}

create_feature_matrix = function(dt, features, h_space_size = h_space_size) {
	# 0-based indices
	row_index = rep(0L:(nrow(dt) - 1L), length(features))
	# note that here we adding `text2vec:::hasher(feature, h_space_size)` - hash offset for this feature
	# this reduces number of collisons because. If we won't apply such scheme - identical values of 
	# different features will be hashed to same value
	col_index = Map(function(fnames) {
		# here we calculate offset for each feature
		# hash name of feature to reduce number of collisions 
		# because for eample if we won't hash value of platform=1 will be hashed to the same as advertiser_id=1
		offset = string_hasher(paste(fnames, collapse = "_"), h_space_size)
		# calculate index = offest + sum(feature values)
		index = (offset + Reduce(`+`, dt[, fnames, with = FALSE])) %% h_space_size
		as.integer(index)
	}, features) %>%
	unlist(recursive = FALSE, use.names = FALSE)

	m = sparseMatrix(i = row_index, j = col_index, x = 1,
				   dims = c(nrow(dt), h_space_size),
				   index1 = FALSE, giveCsparse = FALSE, check = FALSE)
	m
}


### 3-0. prepare-baseline-1 ------------------------------

# source("misc.R")

library("readr")

## import promoted_content export rds
promo = read_csv(sprintf("%s/promoted_content.csv.zip", RAW_DATA_PATH))
promo = as.data.table(promo)

setnames(promo, 'document_id', 'promo_document_id')
save_rds_compressed(promo, sprintf("%s/promo.rds", RDS_DATA_PATH))
rm(promo)

## import clicks_train / clicks_test export rds
clicks_train = read_csv(sprintf("%s/clicks_train.csv.zip", RAW_DATA_PATH))

clicks_test = read_csv(sprintf("%s/clicks_test.csv.zip", RAW_DATA_PATH))

clicks_test = as.data.table(clicks_test)
clicks_train = as.data.table(clicks_train)

clicks_test[, clicked := NA_integer_]
clicks = rbindlist(list(clicks_train, clicks_test));
rm(clicks_test, clicks_train);
gc()
save_rds_compressed(clicks, sprintf("%s/clicks.rds", RDS_DATA_PATH))

## import events export rds
events = read_csv(sprintf("%s/events.csv.zip", RAW_DATA_PATH))
# several values in "platform" column has som bad values, so we need to remove these rows or convert to some value
events[, platform := as.integer(platform)]
# I chose to convert them to most common value
events[is.na(platform), platform := 1L]

events[, uuid := string_hasher(uuid)] # text2vec package install 해야함

geo3 = strsplit(events$geo_location, ">", T) %>% lapply(function(x) x[1:3]) %>% simplify2array(higher = FALSE)

events[, geo_location := string_hasher(geo_location)]
events[, country := string_hasher(geo3[1,])]
events[, state := string_hasher(geo3[2,])]
events[, dma := string_hasher(geo3[3,])]
rm(geo3);
gc()
events[, train := display_id %in% unique(clicks[!is.na(clicked), display_id])]

events[, day := as.integer((timestamp / 1000) / 60 / 60 / 24)]

set.seed(1L)
events[, cv := TRUE]

# leave 11-12 days for validation as well as 15% of events in days 1-10
events[day <= 10, cv := sample(c(FALSE, TRUE), .N, prob = c(0.85, 0.15), replace = TRUE), by = day]

# sort by uuid - not imoprtant at this point. Why we are doing this will be explained below.
setkey(events, uuid)

# save events for future usage
save_rds_compressed(events, sprintf("%s/events.rds", RDS_DATA_PATH))
rm(clicks,events)

### 3-1. prepare-baseline-1 ------------------------------
# this takes ~ 20 minutes (this part is single threaded)

# source("misc.R")

events = readRDS(sprintf("%s/events.rds", RDS_DATA_PATH))
clicks = readRDS(sprintf("%s/clicks.rds", RDS_DATA_PATH))
promo = readRDS(sprintf("%s/promo.rds", RDS_DATA_PATH))

interactions = c('promo_document_id', 'campaign_id', 'advertiser_id', 'document_id', 'platform', 'country', 'state') %>%
    combn(2, simplify = FALSE)

single_features = c('ad_id', 'campaign_id', 'advertiser_id', 'document_id', 'platform', 'geo_location', 'country', 'state', 'dma')

features_with_interactions = c(single_features, interactions)

for (i in 0L:(N_PART - 1L)) {

	dt_chunk = events[uuid %% N_PART == i]
	dt_chunk = clicks[dt_chunk, on = .(display_id = display_id)]
	dt_chunk = promo[dt_chunk, on = .(ad_id = ad_id)]
	setkey(dt_chunk, uuid, display_id, ad_id)

	# TRAIN
	dt_temp = dt_chunk[!is.na(clicked) & cv == FALSE]
	X = create_feature_matrix(dt_temp, features_with_interactions, events_h_size)
	chunk = list(X = X, y = dt_temp$clicked, dt = dt_temp[, .(uuid, document_id, promo_document_id, campaign_id, advertiser_id, display_id, ad_id)])
	save_rds_compressed(chunk, sprintf("%s/%03d.rds", RDS_BASELINE_1_MATRIX_DIR_TRAIN, i))
	# CV
	dt_temp = dt_chunk[!is.na(clicked) & cv == TRUE]
	X = create_feature_matrix(dt_temp, features_with_interactions, events_h_size)
	chunk = list(X = X, y = dt_temp$clicked, dt = dt_temp[, .(uuid, document_id, promo_document_id, campaign_id, advertiser_id, display_id, ad_id)])
	save_rds_compressed(chunk, sprintf("%s/%03d.rds", RDS_BASELINE_1_MATRIX_DIR_CV, i))
	# TEST
	dt_temp = dt_chunk[is.na(clicked)]
	X = create_feature_matrix(dt_temp, features_with_interactions, events_h_size)
	chunk = list(X = X, y = dt_temp$clicked, dt = dt_temp[, .(uuid, document_id, promo_document_id, campaign_id, advertiser_id, display_id, ad_id)])
	save_rds_compressed(chunk, sprintf("%s/%03d.rds", RDS_BASELINE_1_MATRIX_DIR_TEST, i))

	message(sprintf("%s chunk %03d done", Sys.time(), i))
}

### 3-2. run-baseline-1 ------------------------------
# this takes ~6 minutes on 4 core laptop (4 cores + 4 hyperthreads)

# source("misc.R")

if (!("FTRL" %in% installed.packages()[, "Package"]))
	devtools::install_github("dselivanov/FTRL")

library(FTRL)

## cross vaidation related stuff - pick chunks on which we want 

CV_CHUNKS = c(0L:1L)
cv = lapply(CV_CHUNKS, function(x) readRDS(sprintf("%s/%03d.rds", RDS_BASELINE_1_MATRIX_DIR_CV, x)))
dt_cv = lapply(cv, function(x) x[["dt"]]) %>% rbindlist
y_cv = lapply(cv, function(x) x[["y"]]) %>% do.call(c, .) %>% as.numeric
X_cv = lapply(cv, function(x) x[["X"]]) %>% do.call(rbind, .) %>% as("RsparseMatrix")
rm(cv)

## TUNE hyperparameters on train-cv data  (alpha, beta, lambda, etc)

ftrl = FTRL$new(alpha = 0.05, beta = 0.5, lambda = 1, l1_ratio = 1, dropout = 0)

for (i in 0:(N_PART - 1)) {
	data = readRDS(sprintf("%s/%03d.rds", RDS_BASELINE_1_MATRIX_DIR_TRAIN, i))
	y = as.numeric(data$y)
	# update model
	ftrl$partial_fit(X = data$X, y = y, nthread = N_THREAD_FTRL)

	if (i %% 16 == 0) {
		train_auc = glmnet::auc(y, ftrl$predict(data$X))
		p = ftrl$predict(X_cv)
		dt_cv_copy = copy(dt_cv[, .(display_id, clicked = y_cv, p = -p)])
		setkey(dt_cv_copy, display_id, p)
		mean_map12 = dt_cv_copy[, .(map_12 = 1 / which(clicked == 1)), by = display_id][['map_12']] %>%
  	  mean %>% round(5)
		cv_auc = glmnet::auc(y_cv, p)
		message(sprintf("%s batch %03d train_auc = %.4f, cv_auc = %.4f, map@12 = %.4f", Sys.time(), i, train_auc, cv_auc, mean_map12))
	}
}

""" 
# should see something like:
# 2017 - 11 - 20 17:43:37 batch 000 train_auc = 0.7470, cv_auc = 0.7017, map@12 = 0.6277
# 2017 - 11 - 20 17:44:45 batch 016 train_auc = 0.7617, cv_auc = 0.7260, map@12 = 0.6452
# 2017 - 11 - 20 17:45:49 batch 032 train_auc = 0.7634, cv_auc = 0.7308, map@12 = 0.6492
# 2017 - 11 - 20 17:46:52 batch 048 train_auc = 0.7631, cv_auc = 0.7334, map@12 = 0.6507
# 2017 - 11 - 20 17:47:55 batch 064 train_auc = 0.7657, cv_auc = 0.7347, map@12 = 0.6527
# 2017 - 11 - 20 17:48:57 batch 080 train_auc = 0.7649, cv_auc = 0.7360, map@12 = 0.6538
# 2017 - 11 - 20 17:50:00 batch 096 train_auc = 0.7658, cv_auc = 0.7370, map@12 = 0.6546
# 2017 - 11 - 20 17:51:06 batch 112 train_auc = 0.7678, cv_auc = 0.7377, map@12 = 0.6550
"""

## TRAIN FOR SUBMISSION ON FULL DATA (train + cv)
message(sprintf("%s start train model on full train data (train + cv files)", Sys.time()))

ftrl = FTRL$new(alpha = 0.05, beta = 0.5, lambda = 1, l1_ratio = 1, dropout = 0)

for (dir in c(RDS_BASELINE_1_MATRIX_DIR_TRAIN, RDS_BASELINE_1_MATRIX_DIR_CV)) {
	for (i in 0:(N_PART - 1)) {
		data = readRDS(sprintf("%s/%03d.rds", dir, i))
		y = as.numeric(data$y)
		# update model
		ftrl$partial_fit(X = data$X, y = y, nthread = N_THREAD_FTRL)

		if (i %% 16 == 0) {
			message(sprintf("%s batch %03d of %s done", Sys.time(), i, basename(dir)))
		}
	}
}

message(sprintf("%s training done. Saving model...", Sys.time()))
save_rds_compressed(ftrl$dump(), PATH_MODEL_1)

### 3-3. predict-baseline-1 ------------------------------

# source("misc.R")

library(FTRL)

model_dump = readRDS(PATH_MODEL_1)

ftrl = FTRL$new()

ftrl$load(model_dump);

rm(model_dump)

ad_probabilities = lapply(0:(N_PART - 1), function(i) {
	test_data_chunk = readRDS(sprintf("%s/%03d.rds", RDS_BASELINE_1_MATRIX_DIR_TEST, i))
	X = test_data_chunk$X
	dt = test_data_chunk$dt[, .(display_id, ad_id)]
	rm(test_data_chunk);
	dt[, p := ftrl$predict(X)]
	if (i %% 16 == 0)
		message(sprintf("%s %03d", Sys.time(), i))
	dt
}) %>% rbindlist()

# create p_neg - data.table setkey sort in ascedenting order
ad_probabilities[, p_neg := -p]
setkey(ad_probabilities, display_id, p_neg)

# create submission 
ad_subm = ad_probabilities[, .(ad_id = paste(ad_id, collapse = " ")), keyby = display_id]

fwrite(x = ad_subm, file = PATH_MODEL_1_SUBMISSION_FILE)

### 4-1. prepare-baseline-1 ------------------------------
# this takes ~ 45 minutes on 4 core laptop

# source("misc.R")

events = readRDS(sprintf("%s/events.rds", RDS_DATA_PATH))
uuid_events = unique(events$uuid);
rm(events)

colnames = c("uuid", "document_id", "timestamp", "platform", "geo_location", "traffic_source")

fls = list.files(PAGE_VIEWS_CHUNKS_PATH, full.names = TRUE)

foreach(f = fls, .inorder = F, .combine = cbind, .multicombine = TRUE,
		.packages = c("data.table", "magrittr", "text2vec","readr"),
		.options.multicore = list(preschedule = FALSE)) %do% {
			if (basename(f) == "xaa.gz") header = TRUE else header = FALSE
			# will only need c("uuid", "document_id", "timestamp") -  first 3 columns
			# fread can consume UNIX pipe as input, which is not the thing many people know about
			# dt = fread(paste("zcat < ", f), header = header, col.names = colnames[1:3], select = 1:3, showProgress = FALSE)
			print(f)
			dt = read_csv(f, col_names = FALSE)
			dt = dt[, 1:3]
			colnames(dt) = c("uuid", "document_id", "timestamp")
			dt = as.data.table(dt)

			dt[, uuid := string_hasher(uuid)]
			# filter out not observed uuids
			j = dt[['uuid']] %in% uuid_events
			dt = dt[j,]
			# partition by uuid and save
			for (i in 0L:(N_PART - 1L)) {
				out = sprintf("%s%s.rds", VIEWS_INTERMEDIATE_DIR, basename(f))
				save_rds_compressed(dt[uuid %% N_PART == i,], out)
			}
			print(paste0(f,"_save completed"))
			rm(dt);
			gc();
			message(sprintf("%s chunk %s done", Sys.time(), basename(f)))
		}

# rds 파일로 저장

res = foreach(chunk = 0L:(N_PART - 1L), .inorder = FALSE, .multicombine = TRUE,
			  .options.multicore = list(preschedule = FALSE),
			  .packages = c("data.table", "magrittr")) %dopar% {
				dir = sprintf("%s", VIEWS_INTERMEDIATE_DIR)
				fls = list.files(dir)
				print(f)
				dt = fls %>%
				  lapply(function(f) readRDS(sprintf("%s/%s", dir, f))) %>%
				  rbindlist

				save_rds_compressed(dt, sprintf("%s/%03d.rds", VIEWS_DIR, chunk))
				message(sprintf("%s chunk %03d done", Sys.time(), chunk))
				print(sprintf("%s chunk %03d done", Sys.time(), chunk))
			  }

unlink(VIEWS_INTERMEDIATE_DIR, recursive = TRUE)

