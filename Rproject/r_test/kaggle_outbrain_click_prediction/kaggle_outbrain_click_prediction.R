
kaggle - https://www.kaggle.com/c/outbrain-click-prediction

####_0. Data load and first Prediction export -----------------------
if (!require("data.table")) { install.packages("data.table") }

# load data using fread
clicks_test <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/clicks_test.csv", stringsAsFactors = T)

clicks_train <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/clicks_train.csv", stringsAsFactors = T)

documents_categories <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/documents_categories.csv", stringsAsFactors = T)
documents_entities <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/documents_entities.csv", stringsAsFactors = T)
documents_meta <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/documents_meta.csv", stringsAsFactors = T)
documents_topics <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/documents_topics.csv", stringsAsFactors = T)

events <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/events.csv", stringsAsFactors = T)

page_views_sample <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/page_views_sample.csv", stringsAsFactors = T)

promoted_content <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/promoted_content.csv", stringsAsFactors = T)

sample_submission <- fread("C:/Users/bevis/Downloads/kaggle_Outbrain_Click_Prediction/sample_submission.csv", stringsAsFactors = T)

####_1. Data_description -----------------------
	# 인터넷은 가능성있는 보물을 자극합니다. 매일 우리는 우리 지역 사회와 관련된 뉴스 기사를 우연히 발견하거나 다음 여행지를 다루는 기사를 발견 할 때 뜻밖의 일을 경험합니다. 웹을 선도하는 콘텐츠 검색 플랫폼 인 Outbrain 은 즐겨 찾는 사이트를 검색하는 동안 이러한 순간을 제공합니다.

	# 현재 Outbrain은 관련 콘텐츠를 호기심 많은 독자와 매주 수천 개의 사이트에서 약 2,500 억 개의 맞춤형 권장 사항으로 매칭합니다. 이 경쟁에서 Kagglers는 글로벌 사용자 기반이 클릭 할 가능성이있는 콘텐츠를 예측하는 데 어려움을 겪고 있습니다. Outbrain의 추천 알고리즘을 개선하면 더 많은 사용자가 개별 취향을 충족시키는 스토리를 발견 할 수 있습니다.
### clicks_train -----------------------
	# 클릭 한 광고 세트를 보여주는 train data
##_ list -----------------------
	# display_id
	# ad_id
	# clicked(1 if clicked, 0 otherwise)
### clicks_test -----------------------
	# 클릭 된 광고가없는 것을 제외하고는 clicks_train.csv와 동일
	# 각 display_id에는 하나의 클릭 광고만 있습니다. 
	# 테스트 세트에는 전체 데이터 세트 시간대의 display_ids가 포함됩니다. 
	# 또한 경쟁 업체의 공개 / 비공개 샘플링은 시간을 기반으로하지 않고 일정한 무작위 추출 방식을 사용합니다. 
	# 이러한 샘플링 선택은 참여자가 시간을 앞당겨 볼 수있는 가능성에도 불구하고 의도적이었습니다.
##_ list -----------------------
	# display_id
	# ad_id
### events  -----------------------
# display_id 컨텍스트에 대한 정보를 제공 / click_train과 click_test 모두 포함
### page_views_sample / page_views -----------------------
	# 문서를 방문하는 사용자의 로그입니다. 
	# 디스크 공간을 절약하기 위해 전체 데이터 세트의 타임 스탬프는 데이터 세트에서 처음으로 상대적입니다. 
	# 실제 방문 시간을 복구하려면 타임 스탬프에 1465876799998을 추가하십시오.
##_ list -----------------------
	# uuid : 유니크값 - 9,202,149
	# document_id : 유니크값 - 59,849
	# timestamp(ms since 1970 - 01 - 01 - 1465876799998) : 
	# platform(desktop = 1, mobile = 2, tablet = 3)  : 1 - 4,403,345 / 2 - 4,678,799 / 3 - 917,855
	# geo_location(country > state > DMA)
	# traffic_source(internal = 1, search = 2, social = 3)
### promoted_content ----------------------- 
	# 광고에 대한 세부 정보를 제공합니다. 
##_ list -----------------------
	# ad_id
	# document_id
	# campaign_id
	# advertiser_id
### documents -----------------------
	# 문서의 내용에 관한 정보와 Outbrain의 각 관계에 대한 자신감을 제공합니다. 
	# 예를 들어, entity_id는 개인, 조직 또는 위치를 나타낼 수 있습니다. 
	# documents_entities.csv의 행은 주어진 엔터티가 문서에서 참조되었음을 나타냅니다.
### documents_topics -----------------------
##_ list -----------------------
	# document_id : 유니크값 - 2,495,423
	# topic_id
	# confidence_level
### documents_entities -----------------------
##_ list -----------------------
	#
### documents_categories -----------------------
##_ list -----------------------
	# document_id : 유니크값 - 2,828,649
	# category_id
	# confidence_level
### documents_meta -----------------------
	# 문서에 대한 세부 정보를 제공
##_ list -----------------------
	# document_id : 유니크값 - 2,999,334
	# source_id(the part of the site on which the document is displayed, e.g. edition.cnn.com)
	# publisher_id
	# publish_time
### sample_submission : 제출 형식 예제

## consol test -----------------
str(documents_categories)

summary(events)

length(unique(page_views_sample$uuid))

head(page_views_sample)

table(page_views_sample$geo_location)


####_2. EDA -----------------------
### events  ----------------------
 # display_id : 유니크값 - 23,120,126
 # uuid : 유니크값 - 19,794,967
 # document_id  : 유니크값 - 894,060
 # timestamp
 # platform(desktop = 1, mobile = 2, tablet = 3) : 1 - 9,027,268 / 2 - 10,976,278 / 3 - 3,116,575 / \\N - 5
 # geo_location(country > state > DMA)

## display_id 유니크값 파악
length(unique(events$display_id)) # 23,120,126

length(unique(clicks_train$display_id)) # 16,874,593 vs nrow 87,141,731
length(unique(clicks_test$display_id)) # 6,245,533 vs nrow 32,225,162
# -> train + test(display_id) = event (display_id)

## uuid_id 유니크값 파악
length(unique(events$uuid)) # 19,794,967 
# -> 1명이 여러 display를 클릭한 경우도 있음

# train과 test의 PK는 display_id로, event에 display_id를 train과 test로 나눠서 파악하여, train과 est의 사용자 수를 나눠서 파악이 필요

event_table <- data.table(events[,1:3])

setkey(event_table, display_id)
setkey(clicks_test, display_id)
setkey(clicks_train, display_id)

# test display_id 기준 uuid, document_id merge
system.time(test_merge_display_id <- clicks_test[event_table,])

length(unique(test_merge_display_id$uuid)) # 

# train display_id 기준 uuid, document_id merge
system.time(train_merge_display_id <- clicks_train[event_table,])

length(unique(train_merge_display_id$uuid)) # 
