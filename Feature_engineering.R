# Dimensionality reduction and feature selection 04.14.2021

# Install necessary packages
# install.packages("ggcorrplot")

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)


# load full df
df <- read.csv("Bitcoin_Fraud_Identification/data/Bitcoin_Full_df.csv")


## We start by looking at correlation (Pearson and Spearman) ##

# convert class "unknown" to 3
levels(df$class) <- sub("unknown", 3, levels(df$class))

# First, explore the correlation between Local Features
df_local <- df %>% select(4:96)

# Pearson correlation
pearson_cor = round(cor(df_local, method = c("pearson")), 2)

heatmap(x = pearson_cor, col = viridis(93))

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


