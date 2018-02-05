if (!require("RSelenium")) { install.packages("RSelenium") }
if (!require("XML")) { install.packages("XML") }


url <- "http://m.speconomy.com/news/articleView.html?idxno=90080"

sel = rsDriver(port = sample(2000:8000, size = 1), browser = "chrome")

sel_client = sel$client
sel_client$navigate(url)

sel$client$executeScript("window.scrollTo(0, document.body.scrollHeight)")

source = sel_client$getPageSource()
xml_list = xmlToList(xmlParse(source, asText = TRUE))

pic_main_unlist = unlist(xml_list$body$div$div$div$section$article$div$div)

df = data.frame(obs = 1:length(pic_main_unlist),
                    value = pic_main_unlist,
                    stringsAsFactors = FALSE)

Encoding(df$value) = "UTF-8"
df = df[grep("[°¡-ÆR]", df$value), 2]

if (!require("dplyr")) { install.packages("dplyr") }



colnames(df) <- "value"

if (!require("KoNLP")) { install.packages("KoNLP") }

#ÇÑ±Û ¼¼Á¾»çÀü ÀÌ¿ë
useSejongDic()

# sapply() ¸®½ºÆ® ÇüÅÂ°¡ ¾Æ´Ñ Çà·Ä ¶Ç´Â º¤ÅÍ ÇüÅÂ·Î °á°ú °ªÀ» ¹ÝÈ¯ÇÏ´Â ÇÔ¼ö
# ÀÐ¾îµå¸° ÅØ½ºÆ® ÆÄÀÏ¿¡¼­ ¸í»ç¸¸ ÃßÃâ
nouns <- sapply(df, extractNoun, USE.NAMES = F)

# ÃßÃâµÈ ¸í»ç »óÀ§ 30°³, ÇÏÀ§ 30°³ È®ÀÎ
# unlist() °á°ú¸¦ ¹éÅÍ Çü½ÄÀ¸·Î ¹ÝÈ¯ÇÏ´Â ÇÔ¼ö
head(unlist(nouns), 30)
tail(unlist(nouns), 30)

# µÎ±ÛÀÚ ÀÌ»óÀÎ ¸í»ç¸¸ ÃßÃâ
# nouns2 ¿¡ ÀÓ½Ã·Î ÀúÀåÇÏ°í ´Ù½Ã nouns¿¡ ÀúÀå
nouns2 <- unlist(nouns)
nouns <- Filter(function(x) { nchar(x) >= 2 }, nouns2)

nouns <- table(nouns)
nouns <- nouns[order(nouns, decreasing = T)]



##### --- Âü°í
if (!require("tm")) { install.packages("tm") }

doc <- Corpus(VectorSource(df)) #2.¸»¹¶Ä¡º¯È¯

doc <- TermDocumentMatrix(doc) #3.TermDocumentMatrixº¯È¯
                          control = list(#¾Æ·¡·Î¿É¼ÇÀ»³ª¿­
                          tokenize = words, #¹Ì¸®¸¸µé¾îµÐÇÔ¼ö(º¸Åë¸í»çÃßÃâ)·Î¹®ÀåÀ»ÀÚ¸§
                          removeNumbers = T, #¼ýÀÚÁ¦°Å
                          removePunctuation = T)) #¹®ÀåºÎÈ£Á¦°Å

Encoding(doc$dimnames$Terms) = "UTF-8"

doc <- as.matrix(doc)
#4.Matrix·Îº¯È¯

doc <- rowSums(doc)
#5.ÇàÀÇÇÕ(ÃÑºóµµ)

doc <- doc[order(doc, decreasing = T)]
#ºóµµ¿ª¼øÁ¤·Ä

as.data.frame(doc[1:20])
#»óÀ§20°³´Ü¾îº¸±â