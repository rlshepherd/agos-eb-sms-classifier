library(tm)
# DATA
dat <- read.csv("data/2016-23-02_tagged_reports.csv",
                 stringsAsFactors = FALSE)
dat$keyword <- as.factor(dat$keyword)
table(dat$keyword)


sms_corpus <- Corpus(VectorSource(dat$remarks))
corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, content_transformer(removeNumbers))
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

sms_dtm <- DocumentTermMatrix(corpus_clean)
sms_raw_train <- dat[1:500,]
sms_raw_test <- dat[501:dim(dat)[1],]
sms_dtm_train <- sms_dtm[1:500,]
sms_dtm_test <- sms_dtm[501:dim(dat)[1],]

sms_corpus_train <- corpus_clean[1:500]
sms_corpus_test <- corpus_clean[501:dim(dat)[1]]

prop.tables <- as.data.frame(cbind(prop.table(table(sms_raw_train$keyword)),
prop.table(table(sms_raw_test$keyword))))

names(prop.tables) <- c("Train", "Test")
prop.tables$diff <- prop.tables$Train - prop.tables$Test
write.csv(prop.tables, file = "prop_tables.csv")

library("wordcloud")
wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)
rain <- subset(sms_raw_train, keyword == "RAIN")
flood <- subset(sms_raw_train, keyword == "FLOOD")
wordcloud(rain$remarks, max.words = 10, scale = c(3, 0.5))
wordcloud(flood$remarks, max.words = 10, scale = c(3, 0.5))


sms_dict <- c(findFreqTerms(sms_dtm_train, 5))
sms_train <- DocumentTermMatrix(sms_corpus_train,
                list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test,
                list(dictionary = sms_dict))

# NAIVE BAYES
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0:1), labels = c("No", "Yes"))
  return(x)
}

sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)

library(e1071)
sms_classifier <- naiveBayes(sms_train, as.factor(tolower(sms_raw_train$keyword)))
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)

x <- CrossTable(sms_test_pred, sms_raw_test$keyword,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted','actual'))

x$t

# RANDOM FOREST

# SVM