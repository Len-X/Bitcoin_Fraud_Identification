# Dimensionality reduction and feature selection 04.14.2021

# Install necessary packages
# install.packages("ggcorrplot")
# install.packages("Boruta")

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(viridisLite)
library(Boruta)


# load full df
df <- read.csv("Bitcoin_Fraud_Identification/data/Bitcoin_Full_df.csv")

attach(df)

## We start by looking at correlation (Pearson and Spearman) ##

# convert class "unknown" to 3
levels(df$class) <- sub("unknown", 3, levels(df$class))

# First, explore the correlation between Local Features
df_local <- df %>% select(4:96)

# Pearson correlation
pearson_cor = round(cor(df_local, method = c("pearson")), 2)

# heatmap(x = pearson_cor, col = viridis(93))

pearson_cor_heatmap <- ggcorrplot(pearson_cor, type = "full",
                                  lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Pearson Correlation Matrix of Local features") +
  theme(plot.title = element_text(hjust=0.5))

pearson_cor_heatmap

# Spearman correlation

spearman_cor = round(cor(df_local, method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor, type = "full",
                                  lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of Local features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap


### Train / Validation / Test Split ###

# full df with all features
# train_full <- df[TimeStep <= 29, ]
train_full <- df %>% filter(TimeStep <= 29)
valid_full <- df %>% filter(TimeStep > 29 & TimeStep <= 39)
test_full <- df %>% filter(TimeStep > 39)

# df with local features
train_local <- train_full %>% select(1:96)
valid_local <- valid_full %>% select(1:96)
test_local <- test_full %>% select(1:96)


### Feature Selection ###

# Boruta is a feature ranking and selection algorithm based on random 
# forests algorithm.
 
# The advantage with Boruta is that it clearly decides if a variable is 
# important or not and helps to select variables that are statistically 
# significant. Besides, you can adjust the strictness of the algorithm by 
# adjusting the p values that defaults to 0.01 and the maxRuns.

# we first perform Boruta algorithm on Local features

set.seed(2021)
boruta_local <-
  df %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local")) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_local)

# maxRuns - maximal number of importance source runs. Default = 100
# doTrace - verbosity level

boruta_all <-
  df %>% 
  filter(class != 3) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_all)


boruta_aggregated <-
  train_full %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Aggregated")) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_aggregated)







