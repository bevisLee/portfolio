article <- read.csv("D:/Github/hashtags.csv") # ÃÊ±â ÆÄÀÏ
article <- read.csv("C:/Users/bevis/Documents/Visual Studio 2017/Projects/R_project/r_test/r_test/article.csv")
## 10890ºÎÅÍ body_hashtag Æ®·¹ÀÌ´× ½ÃÀÛ ÇØ¾ßÇÔ

if (!require("KoNLP")) { install.packages("KoNLP") }

#ÇÑ±Û ¼¼Á¾»çÀü ÀÌ¿ë
useSejongDic()

article$title_hashtag <- NA

for (i in 1:nrow(article)) {
    # sapply() ¸®½ºÆ® ÇüÅÂ°¡ ¾Æ´Ñ Çà·Ä ¶Ç´Â º¤ÅÍ ÇüÅÂ·Î °á°ú °ªÀ» ¹ÝÈ¯ÇÏ´Â ÇÔ¼ö
    # ÀÐ¾îµå¸° ÅØ½ºÆ® ÆÄÀÏ¿¡¼­ ¸í»ç¸¸ ÃßÃâ
    df <- article[i, 1]
    df <- as.character(df)

    nouns <- sapply(df, extractNoun, USE.NAMES = F)

    # µÎ±ÛÀÚ ÀÌ»óÀÎ ¸í»ç¸¸ ÃßÃâ
    # nouns2 ¿¡ ÀÓ½Ã·Î ÀúÀåÇÏ°í ´Ù½Ã nouns¿¡ ÀúÀå
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
    df = df[grep("[°¡-ÆR]", df$value), 2]

    # sapply() ¸®½ºÆ® ÇüÅÂ°¡ ¾Æ´Ñ Çà·Ä ¶Ç´Â º¤ÅÍ ÇüÅÂ·Î °á°ú °ªÀ» ¹ÝÈ¯ÇÏ´Â ÇÔ¼ö
    # ÀÐ¾îµå¸° ÅØ½ºÆ® ÆÄÀÏ¿¡¼­ ¸í»ç¸¸ ÃßÃâ
    df <- as.character(df)

    nouns <- sapply(df, extractNoun, USE.NAMES = F)

    # µÎ±ÛÀÚ ÀÌ»óÀÎ ¸í»ç¸¸ ÃßÃâ
    # nouns2 ¿¡ ÀÓ½Ã·Î ÀúÀåÇÏ°í ´Ù½Ã nouns¿¡ ÀúÀå
    nouns2 <- unlist(nouns)
    nouns <- Filter(function(x) { nchar(x) >= 2 }, nouns2)

    nouns <- as.data.frame(table(nouns))
    nouns <- nouns[c(order(nouns$Freq,decreasing = TRUE)),]

    ifelse(is.null(nrow(nouns)),nouns_all<-NA,
                   nouns_all <- paste(nouns[1, 1], nouns[2, 1], nouns[3, 1], nouns[4, 1], sep = ","))
    

    article[i, 5] <- nouns_all

}

write.csv(article,"article.csv")