# Data Transformation and Sampling 05.29.2021

library(tidyverse)
library(ggplot2)
library(MASS)


### Box-Cox Transformation ###


# load full df
df <- read.csv("Bitcoin_Fraud_Identification/Data/Bitcoin_Full_df.csv")
attach(df)

# convert class "unknown" to 3
levels(df$class) <- sub("unknown", 3, levels(df$class))

### Train / Validation / Test Split ###

set.seed(2021)

# full df with all features
train_full <- df %>% filter(TimeStep <= 29)
valid_full <- df %>% filter(TimeStep > 29 & TimeStep <= 39)
test_full <- df %>% filter(TimeStep > 39)

# df with local features
train_local <- train_full %>% select(1:96)
valid_local <- valid_full %>% select(1:96)
test_local <- test_full %>% select(1:96)



