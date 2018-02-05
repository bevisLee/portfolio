

    kaggle - https://www.kaggle.com/c/outbrain-click-prediction

    ####_0. Data load and first Prediction export -----------------------
    if (!require("data.table")) { install.packages("data.table") }
    # test&train -----------------------
    clicks_test <- fread("E:/kaggle_Outbrain_Click_Prediction/clicks_test.csv", stringsAsFactors = T)
    clicks_train <- fread("E:/kaggle_Outbrain_Click_Prediction/clicks_train.csv", stringsAsFactors = T)
    # documents -----------------------
    documents_categories <- fread("E:/kaggle_Outbrain_Click_Prediction/documents_categories.csv", stringsAsFactors = T)
    documents_entities <- fread("E:/kaggle_Outbrain_Click_Prediction/documents_entities.csv", stringsAsFactors = T)
    documents_meta <- fread("E:/kaggle_Outbrain_Click_Prediction/documents_meta.csv", stringsAsFactors = T)
    documents_topics <- fread("E:/kaggle_Outbrain_Click_Prediction/documents_topics.csv", stringsAsFactors = T)
    # events -----------------------
    events <- fread("E:/kaggle_Outbrain_Click_Prediction/events.csv", stringsAsFactors = T)
    # page views -----------------------
    page_views_sample <- fread("E:/kaggle_Outbrain_Click_Prediction/page_views_sample.csv", stringsAsFactors = T)
    page_views <- fread("E:/kaggle_Outbrain_Click_Prediction/page_views.csv", stringsAsFactors = T)
    # promoted content -----------------------
    promoted_content <- fread("E:/kaggle_Outbrain_Click_Prediction/promoted_content.csv", stringsAsFactors = T)
    # sample submission -----------------------
    sample_submission <- fread("E:/kaggle_Outbrain_Click_Prediction/sample_submission.csv", stringsAsFactors = T)

    ####_1. Data_description -----------------------
    # ���ͳ��� ���ɼ��ִ� ������ �ڱ��մϴ�. ���� �츮�� �츮 ���� ��ȸ�� ���õ� ���� ��縦 �쿬�� �߰��ϰų� ���� �������� �ٷ�� ��縦 �߰� �� �� ����� ���� �����մϴ�. ���� �����ϴ� ������ �˻� �÷��� �� Outbrain �� ��� ã�� ����Ʈ�� �˻��ϴ� ���� �̷��� ������ �����մϴ�.

    # ���� Outbrain�� ���� �������� ȣ��� ���� ���ڿ� ���� ��õ ���� ����Ʈ���� �� 2,500 �� ���� ������ ���� �������� ��Ī�մϴ�. �� ���￡�� Kagglers�� �۷ι� ����� ����� Ŭ�� �� ���ɼ����ִ� �������� �����ϴ� �� ������� �ް� �ֽ��ϴ�. Outbrain�� ��õ �˰������� �����ϸ� �� ���� ����ڰ� ���� ������ ������Ű�� ���丮�� �߰� �� �� �ֽ��ϴ�.
    ### clicks_train -----------------------
    # Ŭ�� �� ���� ��Ʈ�� �����ִ� train data
    ##_ list -----------------------
    # display_id
    # ad_id
    # clicked(1 if clicked, 0 otherwise)
    ### clicks_test -----------------------
    # Ŭ�� �� ���������� ���� �����ϰ��� clicks_train.csv�� ����
    # �� display_id���� �ϳ��� Ŭ�� ������ �ֽ��ϴ�. 
    # �׽�Ʈ ��Ʈ���� ��ü ������ ��Ʈ �ð����� display_ids�� ���Ե˴ϴ�. 
    # ���� ���� ��ü�� ���� / ����� ���ø��� �ð��� ����������� �ʰ� ������ ������ ���� ����� ����մϴ�. 
    # �̷��� ���ø� ������ �����ڰ� �ð��� �մ�� �� ���ִ� ���ɼ����� �ұ��ϰ� �ǵ����̾����ϴ�.
    ##_ list -----------------------
    # display_id
    # ad_id
    ### events  -----------------------
    # display_id ���ؽ�Ʈ�� ���� ������ ���� / click_train�� click_test ��� ����
    ### page_views_sample / page_views -----------------------
    # ������ �湮�ϴ� ������� �α��Դϴ�. 
    # ��ũ ������ �����ϱ� ���� ��ü ������ ��Ʈ�� Ÿ�� �������� ������ ��Ʈ���� ó������ ������Դϴ�. 
    # ���� �湮 �ð��� �����Ϸ��� Ÿ�� �������� 1465876799998�� �߰��Ͻʽÿ�.
    ##_ list -----------------------
    # uuid : ����ũ�� - 9,202,149
    # document_id : ����ũ�� - 59,849
    # timestamp(ms since 1970 - 01 - 01 - 1465876799998) : 
    # platform(desktop = 1, mobile = 2, tablet = 3)  : 1 - 4,403,345 / 2 - 4,678,799 / 3 - 917,855
    # geo_location(country > state > DMA)
    # traffic_source(internal = 1, search = 2, social = 3)
    ### promoted_content ----------------------- 
    # ������ ���� ���� ������ �����մϴ�. 
    ##_ list -----------------------
    # ad_id
    # document_id
    # campaign_id
    # advertiser_id
    ### documents -----------------------
    # ������ ���뿡 ���� ������ Outbrain�� �� ���迡 ���� �ڽŰ��� �����մϴ�. 
    # ���� ���, entity_id�� ����, ���� �Ǵ� ��ġ�� ��Ÿ�� �� �ֽ��ϴ�. 
    # documents_entities.csv�� ���� �־��� ����Ƽ�� �������� �����Ǿ����� ��Ÿ���ϴ�.
    ### documents_topics -----------------------
    ##_ list -----------------------
    # document_id : ����ũ�� - 2,495,423
    # topic_id
    # confidence_level
    ### documents_entities -----------------------
    ##_ list -----------------------
    #
    ### documents_categories -----------------------
    ##_ list -----------------------
    # document_id : ����ũ�� - 2,828,649
    # category_id
    # confidence_level
    ### documents_meta -----------------------
    # ������ ���� ���� ������ ����
    ##_ list -----------------------
    # document_id : ����ũ�� - 2,999,334
    # source_id(the part of the site on which the document is displayed, e.g. edition.cnn.com)
    # publisher_id
    # publish_time
    ### sample_submission : ���� ���� ����

    ## consol test -----------------
    str(documents_categories)

    summary(events)

    length(unique(page_views_sample$uuid))

    head(page_views_sample)

    table(page_views_sample$geo_location)

    ####_2. EDA -----------------------
    ### events  ----------------------
    # display_id : ����ũ�� - 23,120,126
    # uuid : ����ũ�� - 19,794,967
    # document_id  : ����ũ�� - 894,060
    # timestamp
    # platform (desktop = 1, mobile = 2, tablet = 3) : 1 - 9,027,268 / 2 - 10,976,278 / 3 - 3,116,575 / \\N - 5
    # geo_location (country > state > DMA)

    ## display_id ����ũ�� �ľ�
    length(unique(events$display_id)) # 23,120,126

    length(unique(clicks_train$display_id)) # 16,874,593 vs nrow 87,141,731
    length(unique(clicks_test$display_id)) # 6,245,533 vs nrow 32,225,162
    # -> train + test(display_id) = event (display_id)

    ## uuid_id ����ũ�� �ľ�
    length(unique(events$uuid)) # 19,794,967 
    # -> 1���� ���� display�� Ŭ���� ��쵵 ����

    # train�� test�� PK�� display_id��, event�� display_id�� train�� test�� ������ �ľ��Ͽ�, train�� est�� ����� ���� ������ �ľ��� �ʿ�

    event_table <- data.table(events[, 1:3])

    # test display_id ���� uuid, document_id merge
    test <- as.data.table(clicks_test)

    test[event_table, uuid := i.uuid, on = 'display_id']
    test[event_table, document_id := i.document_id, on = 'display_id']
    test[events, geo_location := i.geo_location, on = 'display_id']

    head(test)

    length(unique(test$uuid)) # 5,861,229

    table(is.na(test$uuid))

    ## geo ���� ���п� ���� �и��Ͽ� ����
    test <- test %>% separate(geo_location, c("country", "state", "DMA"), ">")


    # train display_id ���� uuid, document_id merge
    train <- as.data.table(clicks_train)

    train[event_table, uuid := i.uuid, on = 'display_id']
    train[event_table, document_id := i.document_id, on = 'display_id']
    train[events, geo_location := i.geo_location, on = 'display_id']

    head(train)

    length(unique(train$uuid)) # 14,814,344

    table(is.na(train$uuid))

    ## geo ���� ���п� ���� �и��Ͽ� ����
    train <- train %>% separate(geo_location, c("country", "state", "DMA"), ">")


    ## geo_location ���п� ���� ������ ���� -> train, test�� ������ ��������? event���� ������ ����
    if (!require("dplyr")) { install.packages("dplyr") }
    if (!require("tidyr")) { install.packages("tidyr") }

    events <- events %>% separate(geo_location, c("country", "state", "DMA"), ">")

    table(events$country)

    table(events$state)

    table(events$DMA)


### test code  ----------------------

head(events)

head(clicks_train)

setkey(clicks_train, display_id)
setkey(events, display_id)

train <- merge(clicks_train, events)

rm(clicks_train, events)

head(train)

head(promoted_content)

train[(train$ad_id == 1)]

promoted_content[(promoted_content$ad_id == 1)]

"""
if (!require("caret")) { install.packages("caret") }

set.seed(222)

inTrain <- createDataPartition(y = train$clicked, p = 0.7, list = FALSE)
train_xy <- train[inTrain,]
validation_xy <- train[-inTrain,]


step(lm(clicked ~ ., data = t1), direction = "both")

t1 <- train[1:10000,]
"""

setkey(train, ad_id, document_id)
setkey(promoted_content, ad_id, document_id)

train2 <- promoted_content[train,]

train2 <- merge(train, promoted_content, all = TRUE)


