if (!require("RSelenium")) { install.packages("RSelenium") }
if (!require("XML")) { install.packages("XML") }
if (!require("stringr")) { install.packages("stringr") }
if (!require("dplyr")) { install.packages("dplyr") }

list <- read.csv("D:/Github/list_0828.csv", encoding = "utf-8")
list <- list %>% mutate_if(is.factor, as.character)

for (i in 298:nrow(list)) {
    keyword = list[i, 1]
    keyword = toupper(unlist(iconv(keyword, "cp949", "UTF-8", toRaw = TRUE)))
    keyword = paste0("%", paste(keyword, collapse = "%"))

    url <- paste("https://www.facebook.com/search/people/?init=quick&q=", keyword, sep = "")

    if (i == 1) {
        sel = rsDriver(port = sample(2000:8000, size = 1), browser = "chrome")
    } else {
        print(paste0("...Scraping [[ ", i, " row : ", round(i / nrow(list) * 100, 1), "% ]] Completed..."))
    }

    sel_client = sel$client
    sel_client$navigate(url)

    sel$client$executeScript("window.scrollTo(0, document.body.scrollHeight)")

    source = sel_client$getPageSource()
    xml_list = xmlToList(xmlParse(source, asText = TRUE))

    facebook11 <- xml_list$body$div[2]$div$div$div[2]$div[2]$div[2]$div$div$div$div$div$div$div$div$div$div$a$.attrs[2]
    facebook12 <- xml_list$body$div[3]$div$div$div[2]$div[2]$div[2]$div$div$div$div$div$div$div$div$div$div$a$.attrs[2]

    ifelse(is.null(facebook11), ifelse(is.null(facebook12), facebook1 <- NA, facebook1 <- facebook12), facebook1 <- facebook11)

    facebook21 <- xml_list$body$div[2]$div$div$div[2]$div[2]$div[2]$div$div$div$div$div$div[2]$div$div$div$div$a$.attrs[2]
    facebook22 <- xml_list$body$div[3]$div$div$div[2]$div[2]$div[2]$div$div$div$div$div$div[2]$div$div$div$div$a$.attrs[2]

    ifelse(is.null(facebook21), ifelse(is.null(facebook22), facebook2 <- NA, facebook2 <- facebook22), facebook2 <- facebook21)

    list_name <- list[i, 1]
    facebook <- cbind(list_name, facebook1, facebook2)

    facebook <- data.frame(facebook)
    colnames(facebook) <- c("list_name", "facebook1", "facebook2")
    colnames(facebook_all) <- c("list_name", "facebook1", "facebook2")

    facebook_all <- rbind(facebook_all, facebook)
    rm(keyword, url, list_name, facebook, facebook1, facebook2, facebook11, facebook12, facebook21, facebook22, xml_list, source)
    Sys.sleep(sample(4:10, size = 1))
}

rownames(facebook_all) <- NULL

facebook_all <- facebook_all[-298:-298,]

facebook_all_list <- facebook_all

# url 전처리

facebook_all_list <- facebook_all_list %>% mutate_if(is.factor, as.character)

for (i in 1:nrow(facebook_all_list)) {
    facebook_all_list[i, 2] <- ifelse(is.na(facebook_all_list[i, 2]), facebook_all_list[i, 2], ifelse(grepl("people", facebook_all_list[i, 2]),
                                   paste("https://www.facebook.com/", str_sub(facebook_all_list[i, 2], max(unlist(gregexpr("/", facebook_all_list[i, 2]))) - 3, str_length(facebook_all_list[i, 2])), sep = ""), facebook_all_list[i, 2]))
    facebook_all_list[i, 3] <- ifelse(is.na(facebook_all_list[i, 3]), facebook_all_list[i, 3], ifelse(grepl("people", facebook_all_list[i, 3]),
                                   paste("https://www.facebook.com/", str_sub(facebook_all_list[i, 3], max(unlist(gregexpr("/", facebook_all_list[i, 3]))) - 3, str_length(facebook_all_list[i, 3])), sep = ""), facebook_all_list[i, 3]))

}

# facebook_all_list$facebook <- gsub('<U+FFFD><U+FFFD>', '',facebook_all_list$facebook) 
write.csv(facebook_all_list, "sns_ymlee.csv")

# 엑셀에서 <U+FFFD><U+FFFD>/ -> profile.php?id= 찾아바꾸기 필요 
