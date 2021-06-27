# Dimensionality reduction and feature selection 04.14.2021

# Install necessary packages
# install.packages("ggcorrplot")
# install.packages("Boruta")
# install.packages('hydroGOF')

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(viridisLite)
library(Boruta)
library(caret)
library(h2o)
library(pROC)
library(hydroGOF) # for mse()


# load full df
df <- read.csv("Bitcoin_Fraud_Identification/Data/Bitcoin_Full_df.csv")

attach(df)

## We start by looking at correlation (Pearson and Spearman) ##

# convert class "unknown" to 3
levels(df$class) <- sub("unknown", 3, levels(df$class))

# workaround
# df$class[df$class == "unknown"] <- 3
# df$class <- as.numeric(factor(df$class))
# df$class <- as.factor(df$class)


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
x_train_local <- train_local %>% 
  filter(class != 3) %>% 
  select(4:96) %>%
  as.matrix()
# x_train_local <- as.matrix(x_train_local)

# relevel to two factor levels instead of three
df_train_local <- train_local %>% 
  filter(class != 3)
df_train_local$class <- factor(df_train_local$class, levels = c(1,2))

# set validation data as matrix
x_valid_local <- valid_local %>% 
  filter(class != 3) %>% 
  select(4:96) %>%
  as.matrix()

# relevel to two factor levels instead of three
df_valid_local <- valid_local %>% 
  filter(class != 3)
df_valid_local$class <- factor(df_valid_local$class, levels = c(1,2))

y_train_local <- df_train_local$class
y_valid_local <- df_valid_local$class


# AE with Down-Sapmpled data
# from "Transformation.R"

# set training data as matrix
x_train_local <- train_lf_down %>% 
  select(-Class) %>%
  as.matrix()

# set validation data as matrix
x_valid_local <- valid_lf %>% 
  select(-class) %>%
  as.matrix()

# set test data as matrix
x_test_local <- test_lf %>% 
  select(-class) %>%
  as.matrix()

# outcome variable
y_train_local <- train_lf_down$Class
y_valid_local <- valid_lf$class
y_test_local <- test_lf$class


set.seed(2021)

# set model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 60, activation = "tanh", input_shape = ncol(x_train_local)) %>%
  layer_dense(units = 20, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 60, activation = "tanh") %>%
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
  validation_data = list(x_valid_local, x_valid_local),
  verbose = 1
)

# 20 features: 100 epoch - loss: 0.04446143
# 20 features: 500 epoch - loss: 0.02665084
# 16 features: 100 epoch - loss: 0.06579638
# 16 features: 500 epoch - loss: 0.03933172
# NEW
# 20 features: 500 epoch - loss: 0.04115077
# 20 features: 1000 epoch - loss: 0.03363048
# train + validation
# 20 features train: 100 epoch - loss: 0.08468533
# 20 features validation: 100 epoch - loss: 1.022135
# 20 features train: 500 epoch - loss: 0.03899754, 0.03601634, 0.04308268, 0.03993318, 0.03441159
# 20 features validation: 500 epoch - loss: 1.037613, 1.192277, 1.07344, 1.058106, 1.011728
# Down-Sampled 20 features train: 500 epoch - loss: 0.02771759, valid loss: 1.276879, test loss:0.9515869


# evaluate the performance of the model
mse_ae_train <- evaluate(model, x_train_local, x_train_local)
mse_ae_train

mse_ae_valid <- evaluate(model, x_valid_local, x_valid_local)
mse_ae_valid

mse_ae_test <- evaluate(model, x_test_local, x_test_local)
mse_ae_test

# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, "bottleneck")$output)
# train prediction
intermediate_output_train <- predict(intermediate_layer_model, x_train_local)
intermediate_output_train
# validation prediction
intermediate_output_valid <- predict(intermediate_layer_model, x_valid_local)
intermediate_output_valid
# test prediction
intermediate_output_test <- predict(intermediate_layer_model, x_test_local)
intermediate_output_test

# -------------------------------------------------------------------------------------------------------

## DO NOT RUN FOR DOWN-SAPMLED DATA##

# combine resulted AE features with class
# take an original train class
df_class1 <- train_local %>% 
  filter(class != 3) %>% 
  select(2)

df_class2 <- valid_local %>% 
  filter(class != 3) %>% 
  select(2)

# relevel to two factor levels instead of three
df_class1$class <- factor(df_class1$class, levels = c(1,2))
df_class2$class <- factor(df_class2$class, levels = c(1,2))

# combine class and AE-derived train features
df_ae_train <- cbind(df_class1, intermediate_output_train)
# combine class and AE-derived validation features
df_ae_valid <- cbind(df_class2, intermediate_output_valid)

# save to csv combined AE hidden layer output
write.csv(df_ae_train, "ae_20_variables_train.csv", row.names = FALSE)
write.csv(df_ae_valid, "ae_20_variables_valid.csv", row.names = FALSE)

# -------------------------------------------------------------------------------------------------------

# for down-sampled data

# combine class and AE-derived train features
df_ae_train <- cbind(y_train_local, intermediate_output_train)
# combine class and AE-derived validation features
df_ae_valid <- cbind(y_valid_local, intermediate_output_valid)
# combine class and AE-derived test features
df_ae_test <- cbind(y_test_local, intermediate_output_test)

# save to csv combined AE hidden layer output
write.csv(df_ae_train, "ae_20_down_train.csv", row.names = FALSE)
write.csv(df_ae_valid, "ae_20_variables_valid_new.csv", row.names = FALSE)
write.csv(df_ae_test, "ae_20_variables_test.csv", row.names = FALSE)



### Autoencoder with H2O

# source: https://hub.packtpub.com/implementing-autoencoders-using-h2o/

library("h2o")
h2o.init()

# load training data set with Local Features only

ae_train_local <-
  train_local %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local"))

outcome_name <- "class"
feature_names <- setdiff(names(ae_train_local), outcome_name)

model=h2o.deeplearning(x=feature_names,
                       training_frame=as.h2o(ae_train_local),
                       hidden=c(55, 20, 55),
                       autoencoder = T,
                       activation="Tanh",
                       epochs = 50)
summary(model)

# h2o.varimp - variable importance
model_var_importance <- model@model$variable_importances
model_var_importance


features=h2o.deepfeatures(model,
                          as.h2o(ae_train_local),
                          layer=3)

features

# model@model$scoring_history$training_mse
# ae_mse <- model@model$scoring_history$training_mse
# plot(sort(ae_mse))

# reconstruction error

ae_anomaly <- h2o.anomaly(model, as.h2o(ae_train_local), per_feature = FALSE) ## try per_feature = TRUE
ae_error <- as.data.frame(ae_anomaly)

# sort and plot the reconstructed MSE. 

# The autoencoder struggles from index 25,500 - 26,000 onwards as the error count accelerates upwards. 
# We can determine that the model recognizes patterns in the first roughly 26000 observations that it 
#ncan’t see as easily in the last 300-800.

plot(sort(ae_error$Reconstruction.MSE), main='Reconstruction Error')

# not sure if this is correct!!!
d=as.matrix(features[1:93,])
labels=as.vector(colnames(ae_train_local[-1]))
plot(d,pch=17)
text(d,labels,pos=3)

# Create df with 20 and 55 important (hidden) features

ae_features_20 <- as.data.frame(features) # hidden layer
ae_features_55 <- as.data.frame(features) # last layer

# get the average activation across neurons
features %>% 
  as.data.frame() %>% 
  tidyr::gather() %>%
  summarize(average_activation = mean(value))

#   average_activation
# 1        -0.06564738


# Print performance
# h2o.performance(as.h2o(features), valid = TRUE)



### Recursive Feature Elimination (RFE) algorithm ###

# A simple backwards selection, a.k.a. recursive feature selection (RFE), algorithm

# RFE is in library Caret
library(caret)

set.seed(2021)


rfe_train_local <-
  train_local %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local"))

rfe_train_local <- droplevels(rfe_train_local)

# set 10-fold CV coltrols
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

rfe_local <- rfe(rfe_train_local[,2:94], 
                 rfe_train_local[,1], 
                 sizes=c(2:20), 
                 rfeControl=control)

rfe_local

# list the chosen features
predictors(rfe_local)
# plot the results
plot(rfe_local, type=c("g", "o"), main="Feature Selection with RFE")

# rfe variables
rfe_variables <- as.data.frame(rfe_local$variables)

# save to csv
# write.csv(rfe_variables,"~/Desktop/MASTERS/Bitcoin/rfe_variables.csv", row.names = FALSE)


### Learning Vector Quantization algorithm (LVQ) ###

# (LVQ) is an artificial neural network algorithm that lets you choose how many
# training instances to hang onto and learns exactly what those instances should look like.

lvq_train_local <-
  train_local %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local"))

lvq_train_local <- droplevels(lvq_train_local)

# prepare training scheme
control_lvq <- trainControl(method="repeatedcv", number=10, repeats=3)

# train the model
lvq_model <- train(class ~., data=lvq_train_local, 
                   method="lvq", 
                   preProcess="scale", 
                   trControl=control_lvq)
lvq_model

# estimate variable importance
lvq_importance <- varImp(lvq_model, scale=FALSE)

# summarize importance
# Rank Features By Importance
print(lvq_importance)

# plot importance
plot(lvq_importance, main="Feature Selection with LVQ")

# sort all features by rank of importance
lvq_sorted <- as.data.frame(lvq_importance$importance)
lvq_sorted <- lvq_sorted[order(-lvq_sorted$X1),]
print(lvq_sorted)

# save to csv
# write.csv(lvq_sorted,"~/Desktop/MASTERS/Bitcoin/lvq_sorted_variables.csv", row.names = FALSE)



### Remove highly Correlates Features with Spearman Correlation ###

# All Local Features

train_local_all <-
  train_local %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local"))

spearman_cor_local = round(cor(train_local_all %>% select(!class), method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor_local, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of Local features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
lf_to_remove <- findCorrelation(spearman_cor_local, cutoff = 0.9, names=TRUE)  # 53 features
df_lf <- train_local_all %>% select(!(lf_to_remove))
df_lf_valid <- valid_local %>% select(!(lf_to_remove))



## RFE 16 features

rfe_features <- c("class", "Local_2", "Local_53", "Local_3", "Local_55", "Local_71", "Local_73",
                  "Local_8", "Local_80", "Local_47", "Local_41", "Local_72", "Local_49",
                  "Local_52", "Local_43", "Local_18", "Local_58")

df_rfe <- train_local[, rfe_features]
df_rfe_valid <- valid_local[, rfe_features]

# Spearman Correlation
spearman_cor_rfe = round(cor(df_rfe %>% select(!class), method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor_rfe, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of RFE features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
rfe_to_remove <- findCorrelation(spearman_cor_rfe, cutoff = 0.9, names=TRUE)

df_rfe <- df_rfe %>% select(!(rfe_to_remove))
# df_rfe[, (names(df_rfe) %in% rfe_to_remove)]  # other way
df_rfe_valid <- df_rfe_valid %>% select(!(rfe_to_remove))

# save to csv
# write.csv(df_rfe,"~/Desktop/MASTERS/Bitcoin_Fraud_Identification/Data/rfe_features.csv", row.names = FALSE)


## LVQ 20 features

lvq_features <- c("class", "Local_53", "Local_55", "Local_90", "Local_60", "Local_66", "Local_29", 
                  "Local_23", "Local_5", "Local_14", "Local_41", "Local_47", "Local_89",
                  "Local_49", "Local_43", "Local_31", "Local_25", "Local_18", "Local_91",
                  "Local_30", "Local_24")

df_lvq <- train_local[, lvq_features]

# Spearman Correlation
spearman_cor_lvq = round(cor(df_lvq %>% select(!class), method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor_lvq, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of LVQ features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
lvq_to_remove <- findCorrelation(spearman_cor_lvq, cutoff = 0.9, names=TRUE)

df_lvq <- df_lvq %>% select(!(lvq_to_remove))

# save to csv
# write.csv(df_rfe,"~/Desktop/MASTERS/Bitcoin_Fraud_Identification/Data/lvq_features.csv", row.names = FALSE)


## LVQ 30 features

lvq_features_30 <- c("class", "Local_53", "Local_55", "Local_90", "Local_60", "Local_66", "Local_29", 
                  "Local_23", "Local_5", "Local_14", "Local_41", "Local_47", "Local_89",
                  "Local_49", "Local_43", "Local_31", "Local_25", "Local_18", "Local_91",
                  "Local_30", "Local_24", "Local_10", "Local_54", "Local_84", "Local_4",
                  "Local_6", "Local_78", "Local_42", "Local_48", "Local_83", "Local_52")

df_lvq_30 <- train_local[, lvq_features_30]

# Spearman Correlation
spearman_cor_lvq_30 = round(cor(df_lvq_30 %>% select(!class), method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor_lvq_30, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of 30 LVQ features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
lvq_to_remove_30 <- findCorrelation(spearman_cor_lvq_30, cutoff = 0.9, names=TRUE)

df_lvq_30 <- df_lvq_30 %>% select(!(lvq_to_remove_30))

# save to csv
# write.csv(df_lvq_30,"~/Desktop/MASTERS/Bitcoin_Fraud_Identification/Data/30_original_lvq_features.csv", row.names = FALSE)




## Autoencoder 20 features

ae_features <- read.csv("Bitcoin/ae_results/ae_20features_500epoch.csv")

# Spearman Correlation
spearman_cor = round(cor(ae_features, method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of Autoencoder features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
ae_to_remove <- findCorrelation(spearman_cor, cutoff = 0.9, names=TRUE)

# no highly correlated features to remove
# we use the same df

### TRANSFORMED DATA ###

## re-run with NA features removed ##
# use before finding Highly Correlated features

trasf_features_na <- c("Local_5", "Local_14", "Local_82")

df_trasf_lf_short <- down_train_lf %>% select(!trasf_features_na)
df_transf_valid_lf_short <- transf_valid %>% select(!trasf_features_na)


# All Local Features Transformed data

# train_local_all <- down_train_lf # directly from "Transformation.R"
train_local_all <- train_lf_trans # directly from "Transformation.R"

# or load from CSV
# train_local_all <- read.csv("Bitcoin_Fraud_Identification/Data/transformed_train_local_features.csv")
# or with NA features removed! (run Transformation.R)
# train_local_all <- df_trasf_lf_short # from Logistic_Reg.R

spearman_cor_local = round(cor(train_local_all %>% select(!class), method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor_local, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of Transformed Local features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
lf_to_remove <- findCorrelation(spearman_cor_local, cutoff = 0.9, names=TRUE)  # 53 features removed
df_lf_transf <- train_local_all %>% select(!lf_to_remove)
df_lf_transf_valid <- valid_lf_trans %>% select(!lf_to_remove) # from "Transformation.R"
# df_lf_transf_valid <- df_transf_valid_lf_short %>% select(!lf_to_remove) # from above and from Logistic_Reg.R



### Recursive Feature Elimination (RFE) algorithm ###
### On Transformed data with Highly Correlated features removed ###

# RFE is in library Caret
library(caret)

set.seed(2021)

rfe_train_local <- df_lf_transf
# rfe_train_local <- droplevels(rfe_train_local)

# set 5-fold CV controls
control <- rfeControl(functions=rfFuncs, method="cv", number=5)

# run for transformed down-sampled data + CORR
rfe_local <- rfe(rfe_train_local[,1:40], 
                 rfe_train_local[,41], 
                 sizes=c(2:20), 
                 rfeControl=control)

# run for transformed data + CORR
rfe_local <- rfe(rfe_train_local[,2:42], 
                 rfe_train_local[,1], 
                 sizes=c(2:20), 
                 rfeControl=control)

rfe_local

# list the chosen features
predictors(rfe_local)
# plot the results
plot(rfe_local, type=c("g", "o"), main="Feature Selection with RFE of Transformed data")

# rfe variables
rfe_variables <- predictors(rfe_local)
# rfe_variables <- as.data.frame(rfe_local$variables)

# save to csv
# write.csv(rfe_variables,"~/Desktop/MASTERS/Bitcoin/rfe_variables_transf.csv", row.names = FALSE)



### Learning Vector Quantization algorithm (LVQ) ###
### On Transformed data with Highly Correlated features removed ###

lvq_train_local <- df_lf_transf
# Since we got a warning (see below), let us remove Local_15 feature
# lvq_train_local <- df_lf_transf %>% select(!Local_15)

# prepare training scheme
control_lvq <- trainControl(method="repeatedcv", number=5, repeats=3)

# train the model
lvq_model <- train(Class ~., data=lvq_train_local, 
                   method="lvq", 
                   preProcess="scale", 
                   trControl=control_lvq)
lvq_model

# estimate variable importance
lvq_importance <- varImp(lvq_model, scale=FALSE)

# summarize importance
# Rank Features By Importance
print(lvq_importance)

# plot importance
plot(lvq_importance, main="Feature Selection of Transformed data with LVQ")

# sort all features by rank of importance
lvq_sorted <- as.data.frame(lvq_importance$importance)
lvq_sorted <- lvq_sorted[order(-lvq_sorted$X1),]
print(lvq_sorted)

# save to csv
# write.csv(lvq_sorted,"~/Desktop/MASTERS/Bitcoin/lvq_sorted_var_transf_DS.csv", row.names = FALSE)

# Warning in preProcess.default(thresh = 0.95, k = 5, freqCut = 19, uniqueCut = 10,  :
# These variables have zero variances: Local_15
# Since we have this warning, let us remove Local_15 feature



