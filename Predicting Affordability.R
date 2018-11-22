library(dplyr)
library(reshape2)
library(ggplot2)
library(purrr)
library(tidyr)

# Import training and testing data
training <- read.csv(file = "HTrainLast.csv", stringsAsFactors = FALSE)
testing <- read.csv(file = "HTestLastNoY.csv", stringsAsFactors = FALSE)

# Get rid of the observation column 
training = training[, -1]
testing = testing [,-1]

# Exploratory analysis
# Histograms of all numeric columns
training %>% keep(is.numeric) %>%  gather() %>% ggplot(aes(value)) + facet_wrap(~ key, scales = "free") +geom_histogram()



# Scratch work 
training %>% keep(is.character) %>%  gather() %>% ggplot(aes(value)) + facet_wrap(~ key, scales = "free") +geom_boxplot()

# 
ggplot(data = training, aes(x = affordabilitty, y = OverallQual)) +  geom_boxplot() 

