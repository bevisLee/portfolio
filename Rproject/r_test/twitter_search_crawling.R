if (!require("RSelenium")) { install.packages("RSelenium") }
if (!require("XML")) { install.packages("XML") }

list <- read.csv("D:/Github/list4.csv", encoding = "utf-8")

for (j in 1:nrow(list)) {
	keyword = list[j, 1]
	keyword = toupper(unlist(iconv(keyword, "cp949", "UTF-8", toRaw = TRUE)))
	keyword = paste0("%", paste(keyword, collapse = "%"))

	url1 <- paste("https://twitter.com/search?f=users&vertical=default&q=", keyword, sep = "")

	if (j == 1) {
		sel = rsDriver(port = sample(2000:8000, size = 1), browser = "chrome")
	} else {
		print(paste0("...Scraping [[ ", j, " row : ", round(j / nrow(list) * 100, 1), "% ]] Completed..."))
	}

	sel_client = sel$client
	sel_client$navigate(url1)

	sel$client$executeScript("window.scrollTo(0, document.body.scrollHeight)")

	source = sel_client$getPageSource()
	xml_list = xmlToList(xmlParse(source, asText = TRUE))

	twitter1 <- xml_list$body[4]$div[2]$div$div[2]$div$div$div[5]$div$div$div$div[2]$div$div$div$div$div$a$.attrs[2]
	twitter2 <- xml_list$body[4]$div[2]$div$div[2]$div$div$div[5]$div$div$div$div[2]$div$div[2]$div$div$div$a$.attrs[2]

	ifelse(is.null(twitter1), twitter1 <- NA, twitter1 <- paste("https://twitter.com", twitter1, sep = ""))
	ifelse(is.null(twitter2), twitter2 <- NA, twitter2 <- paste("https://twitter.com", twitter2, sep = ""))

	twitter <- cbind(twitter1, twitter2)
	twitter <- data.frame(twitter)
	colnames(twitter) <- c("twitter1", "twitter2")
	colnames(twitter_all) <- c("twitter1", "twitter2")

	twitter_all <- rbind(twitter_all, twitter)
	rm(keyword, url1, twitter, twitter1, twitter2, xml_list, source)
}

if (!require("beepr")) { install.packages("beepr") }
beep(8)

twitter_all_list <- cbind(list, twitter_all)