article <- read.csv("D:/Github/hashtags.csv") # 초기 파일
article <- read.csv("C:/Users/bevis/Documents/Visual Studio 2017/Projects/R_project/r_test/r_test/article.csv")
## 10890부터 body_hashtag 트레이닝 시작 해야함

if (!require("KoNLP")) { install.packages("KoNLP") }

#한글 세종사전 이용
useSejongDic()

article$title_hashtag <- NA

for (i in 1:nrow(article)) {
    # sapply() 리스트 형태가 아닌 행렬 또는 벡터 형태로 결과 값을 반환하는 함수
    # 읽어드린 텍스트 파일에서 명사만 추출
    df <- article[i, 1]
    df <- as.character(df)

    nouns <- sapply(df, extractNoun, USE.NAMES = F)

    # 두글자 이상인 명사만 추출
    # nouns2 에 임시로 저장하고 다시 nouns에 저장
    nouns2 <- unlist(nouns)
    nouns <- Filter(function(x) { nchar(x) >= 2 }, nouns2)

    for (j in 2:length(nouns)) {
        ifelse(j == 2, nouns_all <- paste(nouns[1], nouns[j], sep = ","), nouns_all <- paste(nouns_all, nouns[j], sep = ","))
    }

    article[i,4] <- nouns_all

}

if (!require("RSelenium")) { install.packages("RSelenium") }
if (!require("XML")) { install.packages("XML") }

article$body_hashtag <- NA

for (i in 10890:nrow(article)) {
    url <- article[i, 2]

    if (i == 1) {
        sel = rsDriver(port = sample(2000:8000, size = 1), browser = "chrome")
    } else {
        print(paste0("...Scraping [[ ", i, " row : ", round(i / nrow(article) * 100, 1), "% ]] Completed..."))
    }

    sel_client = sel$client
    sel_client$navigate(url)

    # sel$client$executeScript("window.scrollTo(0, document.body.scrollHeight)")

    source = sel_client$getPageSource()
    xml_list = xmlToList(xmlParse(source, asText = TRUE))

    pic_main_unlist = unlist(xml_list$body[2]$div[3]$div[3]$div$div$div$div)

    df = data.frame(obs = 1:length(pic_main_unlist),
                    value = pic_main_unlist,
                    stringsAsFactors = FALSE)

    Encoding(df$value) = "UTF-8"
    df = df[grep("[가-힣]", df$value), 2]

    # sapply() 리스트 형태가 아닌 행렬 또는 벡터 형태로 결과 값을 반환하는 함수
    # 읽어드린 텍스트 파일에서 명사만 추출
    df <- as.character(df)

    nouns <- sapply(df, extractNoun, USE.NAMES = F)

    # 두글자 이상인 명사만 추출
    # nouns2 에 임시로 저장하고 다시 nouns에 저장
    nouns2 <- unlist(nouns)
    nouns <- Filter(function(x) { nchar(x) >= 2 }, nouns2)

    nouns <- as.data.frame(table(nouns))
    nouns <- nouns[c(order(nouns$Freq,decreasing = TRUE)),]

    ifelse(is.null(nrow(nouns)),nouns_all<-NA,
                   nouns_all <- paste(nouns[1, 1], nouns[2, 1], nouns[3, 1], nouns[4, 1], sep = ","))
    

    article[i, 5] <- nouns_all

}

write.csv(article,"article.csv")