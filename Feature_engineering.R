# Feature selection and dimentionality reduction

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
train_full <- df %>% filter(TimeStep <= 29)
valid_full <- df %>% filter(TimeStep > 29 & TimeStep <= 39)
test_full <- df %>% filter(TimeStep > 39)

# df with local features
train_local <- train_full %>% select(1:96)
valid_local <- valid_full %>% select(1:96)
test_local <- test_full %>% select(1:96)


### Feature Selection ###

# Boruta
# we first perform Boruta algorithm on Local features
set.seed(2021)

boruta_local <-
  train_local %>% 
  filter(class != 3) %>% 
  select(class, starts_with("Local")) %>% 
  Boruta(class ~ ., data=., doTrace=2)

print(boruta_local)

# maxRuns - maximal number of importance source runs. Default = 100

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

### Recursive Feature Elimination (RFE) algorithm ###

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


### Learning Vector Quantization algorithm (LVQ) ###

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


### Remove highly Correlates Features with Spearman Correlation ###

# all Local Features
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
df_rfe_valid <- df_rfe_valid %>% select(!(rfe_to_remove))


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


## Autoencoder 20 Local features

ae_features <- read.csv("ae_20features_500epoch.csv")

# Spearman Correlation
spearman_cor = round(cor(ae_features, method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of Autoencoder features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
ae_to_remove <- findCorrelation(spearman_cor, cutoff = 0.9, names=TRUE)

# no highly correlated features to remove - we use the same df


### TRANSFORMED DATA ###

trasf_features_na <- c("Local_5", "Local_14", "Local_82")

df_trasf_lf_short <- down_train_lf %>% select(!trasf_features_na)
df_transf_valid_lf_short <- transf_valid %>% select(!trasf_features_na)

# All Local Features Transformed data

# train_local_all <- down_train_lf # directly from "Transformation.R"
train_local_all <- train_lf_trans # directly from "Transformation.R"

# or load from CSV
# train_local_all <- read.csv("Bitcoin_Fraud_Identification/Data/transformed_train_local_features.csv")
# train_local_all <- df_trasf_lf_short # from Logistic_Reg.R

# All Local Features Transformed data with Up-Sampling
train_local_all <- up_train_lf # directly from "Transformation.R"

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


### Recursive Feature Elimination (RFE) algorithm ###
### On Transformed data ###

set.seed(2021)

rfe_train_local <- df_lf_transf
# rfe_train_local <- droplevels(rfe_train_local)

# set 10-fold CV controls
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

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


### RFE on ALL data (AF) ###

set.seed(2021)

rfe_train_af <- train_af

# set 10-fold CV controls
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run on All features
rfe_all <- rfe(rfe_train_af[,2:166], 
                 rfe_train_af[,1], 
                 sizes=c(2:20), 
                 rfeControl=control)

rfe_all

# list the chosen features
predictors(rfe_all)
# plot the results
plot(rfe_all, type=c("g", "o"), main="Feature Selection with RFE on All data")

# rfe variables
rfe_variables <- predictors(rfe_all)

# 20 RFE variables
# rfe_20 <- rfe_variables

# remove highly correlated features
train_all <- train_af %>% select(c(class, all_of(rfe_20)))
# train_all <- rfe_variables

spearman_cor_all = round(cor(train_all %>% select(-class), method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor_all, type = "full",
                                   lab_size=1, tl.cex=8, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of 20 RFE on all features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
af_to_remove <- findCorrelation(spearman_cor_all, cutoff = 0.9, names=TRUE)  # 53 features removed
df_af_train <- train_all %>% select(-af_to_remove)
df_af_valid <- valid_af %>% select(colnames(df_af_train))



### Learning Vector Quantization algorithm (LVQ) ###
### On Transformed data with Highly Correlated features removed ###

lvq_train_local <- df_lf_transf

# prepare control
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
plot(lvq_importance, main="Feature Selection of Transformed data with LVQ")

# sort all features by rank of importance
lvq_sorted <- as.data.frame(lvq_importance$importance)
lvq_sorted <- lvq_sorted[order(-lvq_sorted$X1),]
print(lvq_sorted)

# get 20 most important LVQ features
lvq_features_transf <- row.names(lvq_sorted)[1:20]


### LVQ on ALL data (AF) ###
### AF + LVQ + CORR ###

lvq_train_af <- train_af
# lvq_train_af <- droplevels(lvq_train_af)

set.seed(2021)

# set control
control_lvq <- trainControl(method="repeatedcv", number=10, repeats=3)

# train the model
lvq_model <- train(class ~., data=lvq_train_af, 
                   method="lvq",
                   trControl=control_lvq)
lvq_model

# estimate variable importance
lvq_importance <- varImp(lvq_model, scale=FALSE)

# summarize importance
# Rank Features By Importance
print(lvq_importance)

# plot importance
plot(lvq_importance, main="Feature Selection of All data with LVQ")

# sort all features by rank of importance
lvq_sorted <- as.data.frame(lvq_importance$importance)
lvq_sorted <- lvq_sorted[order(-lvq_sorted$X1),]
print(lvq_sorted)

# get 20 most important LVQ features
lvq_features <- row.names(lvq_sorted)[1:20]
lvq_train_features <- train_af[, lvq_features]

# remove highly correlated features (AF+LVQ+CORR)
# Spearman Correlation
spearman_cor = round(cor(lvq_train_features, method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor, type = "full",
                                   lab_size=1, tl.cex=10, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of LVQ on all Features") +
  theme(plot.title = element_text(hjust=0.5))

spearman_cor_heatmap

# remove highly correlated features
lvq_to_remove <- findCorrelation(spearman_cor, cutoff = 0.9, names=TRUE) # (10 features)
train_lvq_corr <- lvq_train_features %>% select(-lvq_to_remove) # (10 features)
# valid_lvq_corr <- lvq_validation_features %>% select(-lvq_to_remove) # (10 features)
valid_lvq_corr <- valid_af[, lvq_features] %>% select(-lvq_to_remove) # (10 features)


### Correlation on All Features ###

features_af <- train_af %>% select(-class) # directly from "Transformation.R"

# Spearman Correlation
spearman_cor = round(cor(features_af, method = c("spearman")), 2)

spearman_cor_heatmap <- ggcorrplot(spearman_cor, type = "full",
                                   lab_size=1, tl.cex=4, tl.srt=90) +
  ggtitle("Spearman Correlation Matrix of All Features") +
  theme(plot.title = element_text(hjust=0.5))
spearman_cor_heatmap

# remove highly correlated features
af_to_remove <- findCorrelation(spearman_cor, cutoff = 0.9, names=TRUE) # (78 features)
train_af_corr <- train_af %>% select(!af_to_remove) # (87 features)
valid_af_corr <- valid_af %>% select(!af_to_remove) # from "Transformation.R"




