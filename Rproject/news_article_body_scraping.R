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
df = df[grep("[��-�R]", df$value), 2]

if (!require("dplyr")) { install.packages("dplyr") }



colnames(df) <- "value"

if (!require("KoNLP")) { install.packages("KoNLP") }

#�ѱ� �������� �̿�
useSejongDic()

# sapply() ����Ʈ ���°� �ƴ� ��� �Ǵ� ���� ���·� ��� ���� ��ȯ�ϴ� �Լ�
# �о�帰 �ؽ�Ʈ ���Ͽ��� ���縸 ����
nouns <- sapply(df, extractNoun, USE.NAMES = F)

# ����� ���� ���� 30��, ���� 30�� Ȯ��
# unlist() ����� ���� �������� ��ȯ�ϴ� �Լ�
head(unlist(nouns), 30)
tail(unlist(nouns), 30)

# �α��� �̻��� ���縸 ����
# nouns2 �� �ӽ÷� �����ϰ� �ٽ� nouns�� ����
nouns2 <- unlist(nouns)
nouns <- Filter(function(x) { nchar(x) >= 2 }, nouns2)

nouns <- table(nouns)
nouns <- nouns[order(nouns, decreasing = T)]



##### --- ����
if (!require("tm")) { install.packages("tm") }

doc <- Corpus(VectorSource(df)) #2.����ġ��ȯ

doc <- TermDocumentMatrix(doc) #3.TermDocumentMatrix��ȯ
                          control = list(#�Ʒ��οɼ�������
                          tokenize = words, #�̸��������Լ�(�����������)�ι������ڸ�
                          removeNumbers = T, #��������
                          removePunctuation = T)) #�����ȣ����

Encoding(doc$dimnames$Terms) = "UTF-8"

doc <- as.matrix(doc)
#4.Matrix�κ�ȯ

doc <- rowSums(doc)
#5.������(�Ѻ�)

doc <- doc[order(doc, decreasing = T)]
#�󵵿�������

as.data.frame(doc[1:20])
#����20���ܾ��