# Mycode
install.packages("tidytext")
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes

# Load required package
library(tidytext)
library(dplyr)
library(stringr)
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")


tfile = readLines('Martin Luther Speech.txt')
tfile

docs <- Corpus(VectorSource(tfile))
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
docs <- tm_map(docs, toSpace, "/")
docs <- tm_map(docs, toSpace, "@")
docs <- tm_map(docs, toSpace, "\\|")

# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
# Remove your own stop word
# specify your stopwords as a character vector
docs <- tm_map(docs, removeWords, c("blabla1", "blabla2")) 
# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Eliminate extra white spaces
docs <- tm_map(docs, stripWhitespace)
# Text stemming
# docs <- tm_map(docs, stemDocument)
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

# Sentimental Analysis
get_sentiments("bing")
get_sentiments("afinn")
get_sentiments("nrc")
unique(get_sentiments("nrc")$sentiment)
unique(get_sentiments("bing")$sentiment)

tokens <- data_frame(text = tfile) %>% unnest_tokens(word, text)
tokens


# Using bing lexicon to categorize to postive and negative
# and to get net positive minus negative score
tokens %>%
  inner_join(get_sentiments("bing")) %>% # pull out only sentiment words
  count(sentiment) # count the # of positive & negative words

# Using nrc lexicon to categorize to multiple sentiments
tokens %>%
  inner_join(get_sentiments("nrc")) %>% # pull out only sentiment words
  count(sentiment)





# Mycode ends

