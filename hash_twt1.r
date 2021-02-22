#load libraries
library(tm)
library(topicmodels)
library(reshape2)
library(ggplot2)
library(wordcloud)
library(pals)
library(quanteda)
library(stm)
library(lda)
library(tidytext)
library(widyr)
library(textclean)


# load file
textdata <- read.csv("~/Downloads/twts.csv")
# load eng stopwords from
english_stopwords <- readLines("https://slcladal.github.io/resources/stopwords_en.txt", encoding = "UTF-8")

# create corpus object
corpus <- Corpus(DataframeSource(textdata))

# Preprocessing chain
processedCorpus <- tm_map(processedCorpus, removeWords, english_stopwords)
processedCorpus <- tm_map(processedCorpus, removePunctuation, preserve_intra_word_dashes = TRUE)
processedCorpus <- tm_map(processedCorpus, stemDocument, language = "en")
processedCorpus <- tm_map(processedCorpus, stripWhitespace)

# compute document term matrix with terms >= minimumFrequency
minimumFrequency <- 5
DTM <- DocumentTermMatrix(processedCorpus, control = list(bounds = list(global = c(minimumFrequency, Inf))))
# have a look at the number of documents and terms in the matrix
dim(DTM)

sel_idx <- slam::row_sums(DTM) > 0
DTM <- DTM[sel_idx, ]
textdata <- textdata[sel_idx, ]

# number of topics
K <- 20
# set random number generator seed
set.seed(9161)
# compute the LDA model, inference via 1000 iterations of Gibbs sampling
topicModel <- LDA(DTM, K, method="Gibbs", control=list(iter = 1000, verbose = 25))

# have a look a some of the results (posterior distributions)
tmResult <- posterior(topicModel)
# format of the resulting object
attributes(tmResult)

nTerms(DTM)              # lengthOfVocab

# topics are probability distribtions over the entire vocabulary
beta <- tmResult$terms   # get beta from results
dim(beta)

rowSums(beta)

nDocs(DTM)

# for every document we have a probaility distribution of its contained topics
theta <- tmResult$topics 
dim(theta) 

rowSums(theta)[1:10]     # rows in theta sum to 1

terms(topicModel, 10)

exampleTermData <- terms(topicModel, 10)
exampleTermData[, 1:10]

top5termsPerTopic <- terms(topicModel, 5)
topicNames <- apply(top5termsPerTopic, 2, paste, collapse=" ")


# visualize topics as word cloud
topicToViz <- 11

# use 'elect', 'vote', 'nup', 'nrm',? chose 'elect'
topicToViz <- grep('elect', topicNames)[1] # Or select a topic by a term contained in its name
# select to 40 most probable terms from the topic by sorting the term-topic-probability vector in decreasing order

top40terms <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:40]
words <- names(top40terms)

# extract the probabilites of each of the 40 terms
probabilities <- sort(tmResult$terms[topicToViz,], decreasing=TRUE)[1:40]

# visualize the terms as wordcloud
mycolors <- brewer.pal(8, "Dark2")
wordcloud(words, probabilities, random.order = FALSE, color = mycolors)

exampleIds <- c(2, 100, 200)
lapply(corpus[exampleIds], as.character)

exampleIds <- c(2, 100, 200, 400, 900, 2000)
print(paste0(exampleIds[1], ": ", substr(content(corpus[[exampleIds[1]]]), 0, 2000), '...'))

print(paste0(exampleIds[2], ": ", substr(content(corpus[[exampleIds[2]]]), 0, 2000), '...'))

print(paste0(exampleIds[3], ": ", substr(content(corpus[[exampleIds[3]]]), 0, 2000), '...'))

N <- length(exampleIds)
# get topic proportions form example documents
topicProportionExamples <- theta[exampleIds,]
colnames(topicProportionExamples) <- topicNames
vizDataFrame <- melt(cbind(data.frame(topicProportionExamples), document = factor(1:N)), variable.name = "topic", id.vars = "document")  
ggplot(data = vizDataFrame, aes(topic, value, fill = document), ylab = "proportion") + 
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  
  coord_flip() +
  facet_wrap(~ document, ncol = N)

# see alpha from previous model
attr(topicModel, "alpha")

topicModel2 <- LDA(DTM, K, method="Gibbs", control=list(iter = 1000, verbose = 25, alpha = 0.2))

tmResult <- posterior(topicModel2)
theta <- tmResult$topics
beta <- tmResult$terms
topicNames <- apply(terms(topicModel2, 5), 2, paste, collapse = " ")  # reset topicnames

# get topic proportions from example documents
topicProportionExamples <- theta[exampleIds,]
colnames(topicProportionExamples) <- topicNames
vizDataFrame <- melt(cbind(data.frame(topicProportionExamples), document = factor(1:N)), variable.name = "topic", id.vars = "document")  
ggplot(data = vizDataFrame, aes(topic, value, fill = document), ylab = "proportion") + 
  geom_bar(stat="identity") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +  
  coord_flip() +
  facet_wrap(~ document, ncol = N)

# The most probable topics in the entire tweet data
topicProportions <- colSums(theta) / nDocs(DTM)  # mean probabilities over all tweets
names(topicProportions) <- topicNames     # assign the tweet topic names we created before
sort(topicProportions, decreasing = TRUE) # show summed proportions in decreased order

soP <- sort(topicProportions, decreasing = TRUE)
paste(round(soP, 5), ":", names(soP))

countsOfPrimaryTopics <- rep(0, K)
names(countsOfPrimaryTopics) <- topicNames
for (i in 1:nDocs(DTM)) {
  topicsPerDoc <- theta[i, ] # select topic distribution for document i
  # get first element position from ordered list
  primaryTopic <- order(topicsPerDoc, decreasing = TRUE)[1] 
  countsOfPrimaryTopics[primaryTopic] <- countsOfPrimaryTopics[primaryTopic] + 1
}
sort(countsOfPrimaryTopics, decreasing = TRUE)

so <- sort(countsOfPrimaryTopics, decreasing = TRUE)
paste(so, ":", names(so))


