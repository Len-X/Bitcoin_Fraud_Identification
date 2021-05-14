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
df <- read.csv("Bitcoin_Fraud_Identification/Data/Bitcoin_Full_df.csv")

attach(df)

## We start by looking at correlation (Pearson and Spearman) ##

# convert class "unknown" to 3
# levels(df$class) <- sub("unknown", 3, levels(df$class))

df$class[df$class == "unknown"] <- 3
df$class <- as.numeric(factor(df$class))
df$class <- as.factor(df$class)


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

set.seed(2021)

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
  train_local %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local")) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_local)

# maxRuns - maximal number of importance source runs. Default = 100
# doTrace - verbosity level

# plot the boruta_local variable importance
plot(boruta_local, xlab = "", xaxt = "n")
lz <- lapply(1:ncol(boruta_local$ImpHistory), function(i)
boruta_local$ImpHistory[is.finite(boruta_local$ImpHistory[,i]),i])
names(lz) <- colnames(boruta_local$ImpHistory)
Labels <- sort(sapply(lz, median))
axis(side = 1,las=2,labels = names(Labels),
at = 1:ncol(boruta_local$ImpHistory), cex.axis = 0.7)

# obtain the list of confirmed attributes
selected_local <- getSelectedAttributes(boruta_local, withTentative = F)
selected_local

# data frame of the final result derived from Boruta
df_boruta_local <- attStats(boruta_local)
print(df_boruta_local)

# run Boruna on all features

boruta_all <-
  df %>% 
  filter(class != 3) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_all)

# plot the boruta_all variable importance
plot(boruta_all, xlab = "", xaxt = "n")
lz <- lapply(1:ncol(boruta_all$ImpHistory), function(i)
  boruta_all$ImpHistory[is.finite(boruta_all$ImpHistory[,i]),i])
names(lz) <- colnames(boruta_all$ImpHistory)
Labels <- sort(sapply(lz, median))
axis(side = 1,las=2,labels = names(Labels),
  at = 1:ncol(boruta_all$ImpHistory), cex.axis = 0.7)

# obtain the list of confirmed attributes
selected_all <- getSelectedAttributes(boruta_all, withTentative = F)
selected_all

# data frame of the final result derived from Boruta
df_boruta_all <- attStats(boruta_all)
print(df_boruta_all)


# run Boruna on Aggregated features

boruta_aggregated <-
  train_full %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Aggregated")) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_aggregated)


# plot the boruta_aggregated variable importance
plot(boruta_aggregated, xlab = "", xaxt = "n")
lz <- lapply(1:ncol(boruta_aggregated$ImpHistory), function(i)
  boruta_aggregated$ImpHistory[is.finite(boruta_aggregated$ImpHistory[,i]),i])
names(lz) <- colnames(boruta_aggregated$ImpHistory)
Labels <- sort(sapply(lz, median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta_aggregated$ImpHistory), cex.axis = 0.7)

# obtain the list of confirmed attributes
selected_aggregated <- getSelectedAttributes(boruta_aggregated, withTentative = F)
selected_aggregated

# data frame of the final result derived from Boruta
df_boruta_aggregated <- attStats(boruta_aggregated)
print(df_boruta_aggregated)


### Autoencoder ###

# autoencoder in Keras
suppressPackageStartupMessages(library(keras))


# set training data as matrix
x_train_local <- as.matrix(train_local[, 4:96])

set.seed(2021)

# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 55, activation = "tanh", input_shape = ncol(x_train_local)) %>%
  layer_dense(units = 16, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 55, activation = "tanh") %>%
  layer_dense(units = ncol(x_train_local))

# view model layers
summary(model)


# compile model
model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

# fit model
model %>% fit(
  x = x_train_local, 
  y = x_train_local, 
  epochs = 500,  # try 100, 500, 1000
  verbose = 1
)

# 20 features: 100 epoch - loss: 0.04446143
# 20 features: 500 epoch - loss: 0.02665084
# 16 features: 100 epoch - loss: 0.06579638
# 16 features: 500 epoch - loss: 0.03933172


# evaluate the performance of the model
mse_ae <- evaluate(model, x_train_local, x_train_local)
mse_ae

# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, "bottleneck")$output)
intermediate_output <- predict(intermediate_layer_model, x_train_local)
intermediate_output

# save to csv AE hidden layer output
write.csv(intermediate_output, "ae_16features_500epoch.csv", row.names = FALSE)







