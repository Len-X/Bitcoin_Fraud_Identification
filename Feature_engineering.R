# Dimensionality reduction 04.14.2021

library(tidyverse)
library(ggplot2)

# load full df
df <- read.csv("Bitcoin/data/Bitcoin_Full_df.csv")

summary(df)
sd(df$Aggregated_10)
boxplot(df$Aggregated_10)
# All local and aggregated features have been normalized.



