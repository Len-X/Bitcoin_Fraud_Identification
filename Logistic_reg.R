# Classification Model   05.15.2021

### Logistic Regression ###

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(pROC)

### Baseline Model ###

# Fit Logistic Regression to All Local Features

# Data Preprocessing
# remove class 3 from the Train data
# x_train <-
#   df_lf 
  ## use df_lf with removed highly correlated features (from Feature-engineering.R)
  ## or train_local for all local features
x_train <-
  train_local %>% 
  filter(class != 3) %>% 
  select(-c(txId, TimeStep))

# relevel to two factor levels instead of three
x_train$class <- factor(x_train$class, levels = c(1,2))

# Validation
x_validation <-
  valid_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

# remove class 3 from the Validation df with Highly Correlated features removed
# (from Feature-engineering.R)
# x_validation <-
#   df_lf_valid %>% 
#   filter(class != 3) %>%
#   select(-c(txId, TimeStep))

# relevel to two factor levels instead of three
x_validation$class <- factor(x_validation$class, levels = c(1,2))

# split validation df into predictor and outcome variables
x_validation_features <- x_validation %>% select(-class) # predictor variables
y_validation_outcome <- x_validation %>% select(class) # outcome


## fit the GLM model (lf - Local Features)

set.seed(2021)

glm_lf <- glm(class ~ ., data=x_train, family=binomial)

summary(glm_lf)

# access coefficients
summary(glm_lf)$coef
# The smallest p-value here is associated with:

# make predictions
lf_glm_probs <- predict(glm_lf, newdata = x_validation_features, type="response")

plot(lf_glm_probs)

# first 10 probabilities for class 2
lf_glm_probs[1:10]

contrasts(x_train$class)
# with 1 for class 2 and 0 for class 1.

# In order to make a predictions we must convert these predicted probabilities into class labels
lf_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
lf_glm_preds[lf_glm_probs >.5 ] = 2 
# transforms to class "2" all of the elements with predicted probability of class 2 exceeds 0.5

# set levels for predictions
lf_glm_preds <- as.factor(lf_glm_preds)

# Classification Matrix
conf_matrix_lf <- confusionMatrix(lf_glm_preds, y_validation_outcome$class, positive = "1")
conf_matrix_lf

# Confusion matrix summary on Validation data
lf_glm_evaluation <- data.frame(conf_matrix_lf$byClass)
lf_glm_evaluation

#           Reference           
# Prediction    1    2
#          1  874 3057
#          2  164 4904

# with Highly correlated features removed
#           Reference
# Prediction    1    2
#          1   89 1189
#          2  949 6772

# False Positive Rate
fp_rate <- 3057 / (3057+4904); fp_rate
1189 / (1189+6772)

# AUC/ROC

# ROC Train
fit_lf <- fitted(glm_lf)
roc_lf_train <- roc(x_train$class, fit_lf)
ggroc(roc_lf_train)
auc(roc_lf_train)
# Area under the curve: 0.8373, 0.5165

# ROC Test
roc_lf_test <- roc(y_validation_outcome$class, lf_glm_probs)
# ROC plot
ggroc(list(train=roc_lf_train, test=roc_lf_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression without Highly Correlated Local features") +
  labs(color = "")
auc(roc_lf_test)
# Area under the curve: 0.729, 0.4682



# Fit Logistic Regression to RFE data

## RFE 16 features
# DO NOT RUN FOR COR!
rfe_features <- c("class", "Local_2", "Local_53", "Local_3", "Local_55", "Local_71", "Local_73",
                  "Local_8", "Local_80", "Local_47", "Local_41", "Local_72", "Local_49",
                  "Local_52", "Local_43", "Local_18", "Local_58")

df_rfe <- train_local[, rfe_features]

# remove class 3 from the RFE df with correlated features removed
rfe_train <-
  df_rfe %>%    ## run 'Feature_engineering.R' to remove highly correlated features!!!
  filter(class != 3)

# relevel to two factor levels instead of three
rfe_train$class <- factor(rfe_train$class, levels = c(1,2))
  
# transform Validation data into the same shape as train data (from 'Feature_engineering.R')
valid_rfe <- valid_local[, rfe_features]
valid_rfe <- valid_rfe %>% select(!(rfe_to_remove))

# remove class 3 from the RFE df
rfe_validation <-
  valid_rfe %>% 
  filter(class != 3)

# relevel to two factor levels instead of three
rfe_validation$class <- factor(rfe_validation$class, levels = c(1,2))

# split rfe_validation df into predictor and outcome variables
rfe_validation_features <- rfe_validation %>% select(-class) # predictor variables
rfe_validation_outcome <- rfe_validation %>% select(class)


# fit the GLM model

set.seed(2021)

glm_rfe <- glm(class ~ ., data=rfe_train, family=binomial)

summary(glm_rfe)

# access coefficients
summary(glm_rfe)$coef
# The smallest p-value here is associated with: Local_53, Local_18 and Local_52

# make predictions
rfe_glm_probs <- predict(glm_rfe, newdata=rfe_validation_features, type="response")

plot(rfe_glm_probs)

# first 10 probabilities for class 2
rfe_glm_probs[1:10]

# assign class 2 to all probabilities with greater or more 0.5
rfe_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
rfe_glm_preds[rfe_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5


## Classification matrix
classif_matrix <- table(rfe_glm_preds, rfe_validation_outcome$class)
classif_matrix

# rfe_glm_preds    1    2          1    2
#             1   10   28     1   10   78
#             2 1028 7933     2 1028 7883

# set levels for predictions
rfe_glm_preds <- as.factor(rfe_glm_preds)
# false positive rate
28 / (28+7933)
78 / (78+7883)

# Validation Metrics
conf_matrix_rfe <- confusionMatrix(rfe_glm_preds, rfe_validation_outcome$class, positive = "1")
conf_matrix_rfe

# check evaluations

sensitivity(classif_matrix)
recall_rfe <- recall(classif_matrix)
recall_rfe
# [1] 0.009633911

# accuracy
(10+7933)/8999
# [1] 0.8826536

specificity(classif_matrix)
# [1] 0.9964829

precision_rfe <- precision(data = rfe_glm_preds, reference = rfe_validation_outcome$class, relevant = "1")
precision_rfe
# [1] 0.2631579

# Detection Prevalence:  1 − specificity
1-specificity(classif_matrix)
# [1] 0.003517146

# F1-score
F1 <- (2 * ((precision_rfe * recall_rfe) / (precision_rfe + recall_rfe)))

# glm model evaluation on Validation data
rfe_glm_evaluation <- data.frame(conf_matrix_rfe$byClass)
rfe_glm_evaluation

# AUC/ROC

# ROC Train
fit_rfe <- fitted(glm_rfe)
roc_rfe_train <- roc(rfe_train$class, fit_rfe)
ggroc(roc_rfe_train)
auc(roc_rfe_train)
# Area under the curve: 0.8952, 0.9028

# ROC Test
roc_rfe_test <- roc(rfe_validation_outcome$class, rfe_glm_probs)
ggroc(list(train=roc_rfe_train, test=roc_rfe_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with RFE features") +
  labs(color = "")
auc(roc_rfe_test)
# Area under the curve: 0.8429, 0.4999



# Fit Logistic Regression to LVQ data 10 features (20 original features)

# Data Preprocessing

# for LVQ 20  features. DO NOT RUN FOR COR!
lvq_features <- c("class", "Local_53", "Local_55", "Local_90", "Local_60", "Local_66", "Local_29", 
                  "Local_23", "Local_5", "Local_14", "Local_41", "Local_47", "Local_89",
                  "Local_49", "Local_43", "Local_31", "Local_25", "Local_18", "Local_91",
                  "Local_30", "Local_24")

df_lvq <- train_local[, lvq_features]

# otherwise run 'Feature_engineering.R' for df_lvq with highly correlated features removed
# remove class 3 from the LVQ df
lvq_train <-
  df_lvq %>% 
  filter(class != 3)

# relevel to two factor levels instead of three
lvq_train$class <- factor(lvq_train$class, levels = c(1,2))

# transform Validation data into the same shape as train data (from 'Feature_engineering.R')
valid_lvq <- valid_local[, lvq_features]
valid_lvq <- valid_lvq %>% select(!(lvq_to_remove))

# remove class 3 from the LVQ df
lvq_validation <-
  valid_lvq %>% 
  filter(class != 3)

# relevel to two factor levels instead of three
lvq_validation$class <- factor(lvq_validation$class, levels = c(1,2))

# split rfe_validation df into predictor and outcome variables
lvq_validation_features <- lvq_validation %>% select(-class) # predictor variables
lvq_validation_outcome <- lvq_validation %>% select(class)


## fit the GLM model

set.seed(2021)

glm_lvq <- glm(class ~ ., data=lvq_train, family=binomial)

summary(glm_lvq)

# access coefficients
summary(glm_lvq)$coef
# The smallest p-value here is associated with: Local_53, Local_18 and Local_52

# make predictions
lvq_glm_probs <- predict(glm_lvq, newdata=lvq_validation_features, type="response")

plot(lvq_glm_probs)

# first 10 probabilities for class 2
lvq_glm_probs[1:10]

contrasts(lvq_train$class)
# the contrasts() function indicates that R has created a dummy variable
# with 1 for class 2 and 0 for class 1.
#   2
# 1 0
# 2 1

# In order to make a predictions we must convert these predicted probabilities into class labels
# assign class 2 to all probabilities with greater or more 0.5
lvq_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
lvq_glm_preds[lvq_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5


# set levels for predictions
lvq_glm_preds <- as.factor(lvq_glm_preds)

# Classification Matrix
conf_matrix_lvq <- confusionMatrix(lvq_glm_preds, lvq_validation_outcome$class, positive = "1")
conf_matrix_lvq

# glm model evaluation on Validation data
lvq_glm_evaluation <- data.frame(conf_matrix_lvq$byClass)
lvq_glm_evaluation

#                Reference
#  Prediction    1    2
#           1  879 3741
#           2  159 4220

# false positive rate
3741 / (3741+4220)

# AUC/ROC

# ROC Train
fit_lvq <- fitted(glm_lvq)
roc_lvq_train <- roc(lvq_train$class, fit_lvq)
ggroc(roc_lvq_train)
auc(roc_lvq_train)
# Area under the curve:0.7621

# ROC Test
roc_lvq_test <- roc(lvq_validation_outcome$class, lvq_glm_probs)
ggroc(list(train=roc_lvq_train, test=roc_lvq_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with LVQ features") +
  labs(color = "")
auc(roc_lvq_test)
# Area under the curve:0.6885



# Fit Logistic Regression to LVQ data 14 features (30 original features)

# Data Preprocessing
# remove class 3 from the LVQ df
lvq_train_14 <-
  df_lvq_30 %>% 
  filter(class != 3)

# relevel to two factor levels instead of three
lvq_train_14$class <- factor(lvq_train_14$class, levels = c(1,2))

# transform Validation data into the same shape as train data (from 'Feature_engineering.R')
valid_lvq_14 <- valid_local[, lvq_features_30]
valid_lvq_14 <- valid_lvq_14 %>% select(!(lvq_to_remove_30))

# remove class 3 from the LVQ df
lvq_validation_14 <-
  valid_lvq_14 %>% 
  filter(class != 3)

# relevel to two factor levels instead of three
lvq_validation_14$class <- factor(lvq_validation_14$class, levels = c(1,2))

# split rfe_validation df into predictor and outcome variables
lvq_validation_features_14 <- lvq_validation_14 %>% select(-class) # predictor variables
lvq_validation_outcome_14 <- lvq_validation_14 %>% select(class)


## fit the GLM model

set.seed(2021)

glm_lvq_14 <- glm(class ~ ., data=lvq_train_14, family=binomial)

summary(glm_lvq_14)

# access coefficients
summary(glm_lvq_14)$coef
# The smallest p-value here is associated with:

# make predictions
lvq_glm_probs_14 <- predict(glm_lvq_14, newdata=lvq_validation_features_14, type="response")

plot(lvq_glm_probs_14)

# first 10 probabilities for class 2
lvq_glm_probs_14[1:10]

contrasts(lvq_train_14$class)
# the contrasts() function indicates that R has created a dummy variable
# with 1 for class 2 and 0 for class 1.
#   2
# 1 0
# 2 1

# In order to make a predictions we must convert these predicted probabilities into class labels
# assign class 2 to all probabilities with greater or more 0.5
lvq_glm_preds_14 = rep(1, 8999) # creates a vector of 8,999 class "1" elements
lvq_glm_preds_14[lvq_glm_probs_14 >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5


# set levels for predictions
lvq_glm_preds_14 <- as.factor(lvq_glm_preds_14)

# Classification Matrix
conf_matrix_lvq_14 <- confusionMatrix(lvq_glm_preds_14, lvq_validation_outcome_14$class, positive = "1")
conf_matrix_lvq_14

# Confusion matrix summary
conf_matrix_lvq_14$byClass

# glm model evaluation on Validation data
lvq_glm_evaluation_14 <- data.frame(conf_matrix_lvq_14$byClass)
lvq_glm_evaluation_14


# AUC/ROC

# ROC Train
fit_lvq_14 <- fitted(glm_lvq_14)
roc_lvq_train_14 <- roc(lvq_train_14$class, fit_lvq_14)
ggroc(roc_lvq_train_14)
auc(roc_lvq_train_14)
# Area under the curve:

# ROC Test
roc_lvq_test_14 <- roc(lvq_validation_outcome_14$class, lvq_glm_probs_14)
ggroc(list(train=roc_lvq_train_14, test=roc_lvq_test_14), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with 14 LVQ features") +
  labs(color = "")
auc(roc_lvq_test_14)
# Area under the curve:



## Fit Logistic Regression to Autoencoder data ##


# Data Preprocessing

# load AE train
ae_train <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_train.csv")
# ae_train <- df_ae_train
ae_train$class <- as.factor(ae_train$class)

# load Down Sampled AE train data (do not run for all AE train observations!)
# from "transformation.R"
ae_train <- down_train_ae

# load AE validation
ae_validation <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_valid.csv")
# ae_validation <- df_ae_valid
ae_validation$class<- as.factor(ae_validation$class)

# load Validation AE data from "transformation.R"
ae_validation <- valid_ae

# split ae_validation df into predictor and outcome variables
ae_validation_features <- ae_validation %>% select(-class) # predictor variables
ae_validation_outcome <- ae_validation %>% select(class)

## fit the GLM model

set.seed(2021)

glm_ae <- glm(Class ~ ., data=ae_train, family=binomial)

summary(glm_ae)

# access coefficients
summary(glm_ae)$coef
# The smallest p-value here is associated with:

# make predictions
ae_glm_probs <- predict(glm_ae, newdata=ae_validation_features, type="response")

plot(ae_glm_probs)

# first 10 probabilities
ae_glm_probs[1:10]

ae_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
ae_glm_preds[ae_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
ae_glm_preds <- as.factor(ae_glm_preds)

# Classification Matrix
conf_matrix_ae <- confusionMatrix(ae_glm_preds, ae_validation_outcome$class, positive = "1")
conf_matrix_ae

# glm model evaluation on Validation data
ae_glm_evaluation <- data.frame(conf_matrix_ae$byClass)
ae_glm_evaluation

#           Reference
# Prediction    1    2
#          1  674   75
#          2  364 7886

#           Reference
# Prediction    1    2
#          1  889 2378
#          2  149 5583

# False positive rate
75 /(75 + 7886)
2378 /(2378 + 5583)

# AUC/ROC

# ROC Train
fit_ae <- fitted(glm_ae)
roc_ae_train <- roc(ae_train$Class, fit_ae)
ggroc(roc_ae_train)
auc(roc_ae_train)
# Area under the curve: 0.935, 0.9265

# ROC Test
roc_ae_test <- roc(ae_validation_outcome$class, ae_glm_probs)
ggroc(list(train=roc_ae_train, test=roc_ae_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Down-sampled Autoencoder features") +
  labs(color = "")
auc(roc_ae_test)
# Area under the curve: 0.8919, 0.8777



## Fit Logistic Regression to Transformed data (all Local Features) + DS ##

# Data Preprocessing

# load Transformed Downsampled train data (All local features)
# DO NOT run if running with NA features removed!
down_train <- read.csv("Bitcoin_Fraud_Identification/Data/transformed_train_local_features.csv")
down_train$class<- as.factor(down_train$class)

down_train <- down_train_lf # directly from "Transformation.R"

## re-run with NA features removed ##
# down_train <-  df_trasf_lf_short # from code below 

# OR
# with Highly Correlated features removed!
# from Feature_engineering.R!
down_train <- df_lf_transf 

# load Transformed validation data
transf_valid <- read.csv("Bitcoin_Fraud_Identification/Data/.csv")
transf_valid$class<- as.factor(transf_valid$class)
# transf_valid <- valid_lf_trans # directly from "Transformation.R"

## re-run with NA features removed ##
# transf_valid <- df_transf_valid_lf_short
# OR 
# with Highly Correlated features removed!
# transform Validation data into the same shape as train data
transf_valid <- df_lf_transf_valid # from Feature_engineering.R

# split trasf_valid df into predictor and outcome variables
transf_validation_features <- transf_valid %>% select(-class) # predictor variables
transf_validation_outcome <- transf_valid %>% select(class)


## fit the GLM model

set.seed(2021)

glm_transf <- glm(Class ~ ., data=down_train, family=binomial)

summary(glm_transf)

# access coefficients
summary(glm_transf)$coef
# The smallest p-value here is associated with:

# make predictions
transf_glm_probs <- predict(glm_transf, newdata=transf_validation_features, type="response")

plot(transf_glm_probs)

# first 10 probabilities
transf_glm_probs[1:10]

transf_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
transf_glm_preds[transf_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
transf_glm_preds <- as.factor(transf_glm_preds)

# Classification Matrix
conf_matrix_transf <- confusionMatrix(transf_glm_preds, transf_validation_outcome$class, positive = "1")
conf_matrix_transf

# glm model evaluation on Validation data
transf_glm_evaluation <- data.frame(conf_matrix_transf$byClass)
transf_glm_evaluation

#            Reference
# Prediction    1    2
#          1  714  402
#          2  324 7559

#           Reference       # Highly correlated featured removed
# Prediction    1    2
#          1  728  422
#          2  310 7539

# False positive rate
402/(402+7559)
422/(422+7539)

# AUC/ROC

# ROC Train
fit_transf <- fitted(glm_transf)
roc_transf_train <- roc(down_train$Class, fit_transf)
ggroc(roc_transf_train)
auc(roc_transf_train)
# Area under the curve: 0.985, 0.9745

# ROC Test
roc_transf_test <- roc(transf_validation_outcome$class, transf_glm_probs)
ggroc(list(train=roc_transf_train, test=roc_transf_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Transformed Downsampled and Highly Correlated Local features removed") +
  labs(color = "")
auc(roc_transf_test)
# Area under the curve: 0.8986, 0.9029


## re-run with NA features removed ##
# use before finding Highly Correlated features

trasf_features_na <- c("Local_5", "Local_14", "Local_82")

df_trasf_lf_short <- down_train_lf %>% select(!trasf_features_na)
df_transf_valid_lf_short <- transf_valid %>% select(!trasf_features_na)




## Fit Logistic Regression to Transformed data with RFE ##

# Data Preprocessing

## RFE 20 features
rfe_features <- c("Class", "Local_2",  "Local_55", "Local_49", "Local_8", "Local_58", "Local_90", "Local_67", "Local_31",
                  "Local_3", "Local_16", "Local_73", "Local_52", "Local_28", "Local_18", "Local_4", "Local_40",
                  "Local_19", "Local_85", "Local_79", "Local_80")

# subset with transformed down-sampled data
rfe_train <- down_train_lf[, rfe_features]

# load transformed validation data
rfe_validation_features <- valid_lf_trans[, rfe_features[2:21]] # predictor variables
rfe_validation_outcome <- valid_lf_trans %>% select(class) # outcome variable

## fit the GLM model

set.seed(2021)

glm_rfe <- glm(Class ~ ., data=rfe_train, family=binomial)

summary(glm_rfe)

# access coefficients
summary(glm_rfe)$coef
# The smallest p-value here is associated with:

# make predictions
rfe_glm_probs <- predict(glm_rfe, newdata=rfe_validation_features, type="response")

plot(rfe_glm_probs)

# first 10 probabilities
rfe_glm_probs[1:10]

rfe_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
rfe_glm_preds[rfe_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
rfe_glm_preds <- as.factor(rfe_glm_preds)

# Classification Matrix
conf_matrix_rfe <- confusionMatrix(rfe_glm_preds, rfe_validation_outcome$class, positive = "1")
conf_matrix_rfe

# glm model evaluation on Validation data
rfe_glm_evaluation <- data.frame(conf_matrix_rfe$byClass)
rfe_glm_evaluation

#           Reference
# Prediction    1    2
#          1  685  475
#          2  353 7486

# False positive rate
475/(475+7486)

# AUC/ROC

# ROC Train
fit_rfe <- fitted(glm_rfe)
roc_rfe_train <- roc(rfe_train$Class, fit_rfe)
ggroc(roc_rfe_train)
auc(roc_rfe_train)
# Area under the curve: 0.9661

# ROC Test
roc_rfe_test <- roc(rfe_validation_outcome$class, rfe_glm_probs)
ggroc(list(train=roc_rfe_train, test=roc_rfe_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Transformed RFE features") +
  labs(color = "")
auc(roc_rfe_test)
# Area under the curve: 0.8352



## Fit Logistic Regression to Transformed data with LVQ ##

# Data Preprocessing

# for LVQ top 20 features.
lvq_features <- c("Class", "Local_55", "Local_90", "Local_49", "Local_31", "Local_18", "Local_91",
                  "Local_4", "Local_78", "Local_76", "Local_52", "Local_58", "Local_85",
                  "Local_40", "Local_73", "Local_37", "Local_16", "Local_8", "Local_92",
                  "Local_80", "Local_67")

lvq_train <- down_train_lf[, lvq_features]

# transform Validation data into the same shape as train data
lvq_validation_features <- valid_lf_trans[, lvq_features[-1]] # predictor variables
lvq_validation_outcome <- valid_lf_trans %>% select(class) # outcome

## fit the GLM model

set.seed(2021)

glm_lvq <- glm(Class ~ ., data=lvq_train, family=binomial)

summary(glm_lvq)

# access coefficients
summary(glm_lvq)$coef

# make predictions
lvq_glm_probs <- predict(glm_lvq, newdata=lvq_validation_features, type="response")

plot(lvq_glm_probs)

# first 10 probabilities for class 2
lvq_glm_probs[1:10]

contrasts(lvq_train$Class)

# In order to make a predictions we must convert these predicted probabilities into class labels
# assign class 2 to all probabilities with greater or more 0.5
lvq_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
lvq_glm_preds[lvq_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5


# set levels for predictions
lvq_glm_preds <- as.factor(lvq_glm_preds)

# Classification Matrix
conf_matrix_lvq <- confusionMatrix(lvq_glm_preds, lvq_validation_outcome$class, positive = "1")
conf_matrix_lvq

# glm model evaluation on Validation data
lvq_glm_evaluation <- data.frame(conf_matrix_lvq$byClass)
lvq_glm_evaluation

#                Reference
#  Prediction    1    2
#           1  806  319
#           2  232 7642

# false positive rate
319 / (319 + 7642)

# AUC/ROC

# ROC Train
fit_lvq <- fitted(glm_lvq)
roc_lvq_train <- roc(lvq_train$Class, fit_lvq)
ggroc(roc_lvq_train)
auc(roc_lvq_train)
# Area under the curve: 0.9603

# ROC Test
roc_lvq_test <- roc(lvq_validation_outcome$class, lvq_glm_probs)
ggroc(list(train=roc_lvq_train, test=roc_lvq_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with transformed LVQ features") +
  labs(color = "")
auc(roc_lvq_test)
# Area under the curve: 0.945



## Fit Logistic Regression to DS + Autoencoder ##

# Data Preprocessing

# load AE train
ae_train <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_down_train.csv")
# ae_train <- df_ae_train

# rename all of the variables
varname <- c('V')
# define number of times above "varname" repeats itself
n <- c(21) # 21 features, including class
# replace column names
names(ae_train)[1:ncol(ae_train)] <- unlist(mapply(function(x,y) 
  paste(x, seq(0,y), sep="_"), varname, n))

# rename "V_0" variable to "class"
colnames(ae_train)[1] <- "class"

ae_train$class <- as.factor(ae_train$class)

# load AE validation
ae_validation <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_valid_new.csv")

# rename all of the variables in validation data
# replace column names
names(ae_validation)[1:ncol(ae_validation)] <- unlist(mapply(function(x,y) 
  paste(x, seq(0,y), sep="_"), varname, n))

# rename "V_0" variable to "class"
colnames(ae_validation)[1] <- "class"

ae_validation$class<- as.factor(ae_validation$class)

# split ae_validation df into predictor and outcome variables
ae_validation_features <- ae_validation %>% select(-class) # predictor variables
ae_validation_outcome <- ae_validation %>% select(class)

## fit the GLM model

set.seed(2021)

glm_ae <- glm(class ~ ., data=ae_train, family=binomial)

summary(glm_ae)

# access coefficients
summary(glm_ae)$coef

# make predictions
ae_glm_probs <- predict(glm_ae, newdata=ae_validation_features, type="response")

plot(ae_glm_probs)

# first 10 probabilities
ae_glm_probs[1:10]

ae_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
ae_glm_preds[ae_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
ae_glm_preds <- as.factor(ae_glm_preds)

# Classification Matrix
conf_matrix_ae <- confusionMatrix(ae_glm_preds, ae_validation_outcome$class, positive = "1")
conf_matrix_ae

# glm model evaluation on Validation data
ae_glm_evaluation <- data.frame(conf_matrix_ae$byClass)
ae_glm_evaluation

#           Reference
# Prediction    1    2
#          1  970 2090
#          2  68 5871

# False positive rate
2090 /(2090 + 5871)

# AUC/ROC

# ROC Train
fit_ae <- fitted(glm_ae)
roc_ae_train <- roc(ae_train$class, fit_ae)
ggroc(roc_ae_train)
auc(roc_ae_train)
# Area under the curve: 0.9533

# ROC Test
roc_ae_test <- roc(ae_validation_outcome$class, ae_glm_probs)
ggroc(list(train=roc_ae_train, test=roc_ae_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Down-sampled Autoencoder features") +
  labs(color = "")
auc(roc_ae_test)
# Area under the curve: 0.9003



## Fit Logistic Regression to Transformed data (all Local Features) ##

# first, let us run Logistic Regression on all transformed Local features
train_transf_lf <- train_lf_trans # directly from "Transformation.R"

## re-run with unimportant features removed (UFR) ##
# keep only important features at 0.05 significance level (31 features)
important_features <- c("class", "Local_1", "Local_2", "Local_3", "Local_7", "Local_9", 
                        "Local_11", "Local_13", "Local_17", "Local_18", "Local_19", "Local_22", 
                        "Local_28", "Local_35", "Local_37", "Local_41", "Local_43",
                        "Local_52", "Local_55", "Local_57", "Local_58", "Local_64",
                        "Local_66", "Local_76", "Local_77", "Local_78", "Local_83",
                        "Local_84", "Local_88", "Local_89", "Local_90", "Local_91")

train_transf_lf <- train_lf_trans[, important_features]

# OR
# with Highly Correlated features removed!


# load Transformed validation data
valid_transf_lf <- valid_lf_trans # directly from "Transformation.R"

## re-run with unimportant features removed ##
valid_transf_lf <- valid_lf_trans[, important_features]

# OR 
# with Highly Correlated features removed!
# transform Validation data into the same shape as train data


# split trasf_valid_lf df into predictor and outcome variables
transf_validation_features <- valid_transf_lf %>% select(-class) # predictor variables
transf_validation_outcome <- valid_transf_lf %>% select(class)


## fit the GLM model

set.seed(2021)

glm_transf <- glm(class ~ ., data=train_transf_lf, family=binomial)

summary(glm_transf)

# access coefficients
summary(glm_transf)$coef
# The smallest p-value here is associated with:

# make predictions
transf_glm_probs <- predict(glm_transf, newdata=transf_validation_features, type="response")

plot(transf_glm_probs)

# first 10 probabilities
transf_glm_probs[1:10]

transf_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
transf_glm_preds[transf_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
transf_glm_preds <- as.factor(transf_glm_preds)

# Classification Matrix
conf_matrix_transf <- confusionMatrix(transf_glm_preds, transf_validation_outcome$class, positive = "1")
conf_matrix_transf

# glm model evaluation on Validation data
transf_glm_evaluation <- data.frame(conf_matrix_transf$byClass)
transf_glm_evaluation

#            Reference
# Prediction    1    2
#          1  646   67
#          2  392 7894

#           Reference       # unimportant featured removed
# Prediction    1    2
#          1  590   92
#          2  448 7869

# False positive rate
67/(67+7894)
92/(92+7869)

# AUC/ROC

# ROC Train
fit_transf <- fitted(glm_transf)
roc_transf_train <- roc(train_transf_lf$class, fit_transf)
ggroc(roc_transf_train)
auc(roc_transf_train)
# Area under the curve: 0.9834, 0.976

# ROC Test
roc_transf_test <- roc(transf_validation_outcome$class, transf_glm_probs)
ggroc(list(train=roc_transf_train, test=roc_transf_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Transformed Important Local features") +
  labs(color = "")
auc(roc_transf_test)
# Area under the curve: 0.9158, 0.8805



## Fit Logistic Regression to Transformed data with Highly correlated features removed ##

# first, let us run Logistic Regression on transformed Local features + CORR
# (41 features)
train_transf_lf <- df_lf_transf # directly from "Feature_engineering.R"

## re-run with unimportant features removed (UFR) ##
# keep only important features at 0.05 significance level (28 features)
important_features <- c("class", "Local_2", "Local_4", "Local_7", "Local_8", "Local_10",
                        "Local_12", "Local_16", "Local_18", "Local_19", "Local_21",
                        "Local_28", "Local_29", "Local_43", "Local_52", "Local_53",
                        "Local_54", "Local_56", "Local_68", "Local_71", "Local_74",
                        "Local_76", "Local_78", "Local_79", "Local_87", "Local_88",
                        "Local_90", "Local_91", "Local_93")

train_transf_lf <- train_lf_trans[, important_features]

# load Transformed validation data + CORR
valid_transf_lf <- df_lf_transf_valid # directly from "Feature_engineering.R"

## re-run with unimportant features removed ##
valid_transf_lf <- df_lf_transf_valid[, important_features]

# split trasf_valid_lf df into predictor and outcome variables
transf_validation_features <- valid_transf_lf %>% select(-class) # predictor variables
transf_validation_outcome <- valid_transf_lf %>% select(class)

## fit the GLM model

set.seed(2021)

glm_transf <- glm(class ~ ., data=train_transf_lf, family=binomial)

summary(glm_transf)

# access coefficients
summary(glm_transf)$coef

# make predictions
transf_glm_probs <- predict(glm_transf, newdata=transf_validation_features, type="response")

plot(transf_glm_probs)

# first 10 probabilities
transf_glm_probs[1:10]

transf_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
transf_glm_preds[transf_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
transf_glm_preds <- as.factor(transf_glm_preds)

# Classification Matrix
conf_matrix_transf <- confusionMatrix(transf_glm_preds, transf_validation_outcome$class, positive = "1")
conf_matrix_transf

# glm model evaluation on Validation data
transf_glm_evaluation <- data.frame(conf_matrix_transf$byClass)
transf_glm_evaluation

#            Reference
# Prediction    1    2
#          1  402   57
#          2  636 7904

#           Reference       # unimportant featured removed
# Prediction    1    2
#          1  485   84
#          2  553 7877

# False positive rate
57/(57+7904)
84/(84+7877)

# AUC/ROC

# ROC Train
fit_transf <- fitted(glm_transf)
roc_transf_train <- roc(train_transf_lf$class, fit_transf)
ggroc(roc_transf_train)
auc(roc_transf_train)
# Area under the curve: 0.9716, 0.9694

# ROC Test
roc_transf_test <- roc(transf_validation_outcome$class, transf_glm_probs)
ggroc(list(train=roc_transf_train, test=roc_transf_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Transformed Local features, Highly Correlated and Unimportant features removed") +
  labs(color = "")
auc(roc_transf_test)
# Area under the curve: 0.9081, 0.9112



## Logistic Regression on Transformed data COR + RFE + Important Features ##

# first, let us run Logistic Regression on transformed Local features + CORR + RFE
# (10 features)
train_transf_rfe <- train_lf_trans[, c("class", rfe_variables)] # directly from "Feature_engineering.R"
# all 10 features are important!

# load Transformed validation data + CORR + RFE
valid_transf_rfe <- valid_lf_trans[, c("class", rfe_variables)] # directly from "Feature_engineering.R"

# split trasf_valid_lf df into predictor and outcome variables
transf_validation_features <- valid_transf_rfe %>% select(-class) # predictor variables
transf_validation_outcome <- valid_transf_rfe %>% select(class)

## fit the GLM model

set.seed(2021)

glm_transf <- glm(class ~ ., data=train_transf_rfe, family=binomial)
# all 10 features are very important!

summary(glm_transf)

# access coefficients
summary(glm_transf)$coef

# make predictions
transf_glm_probs <- predict(glm_transf, newdata=transf_validation_features, type="response")

plot(transf_glm_probs)

# first 10 probabilities
transf_glm_probs[1:10]

transf_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
transf_glm_preds[transf_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# set levels for predictions
transf_glm_preds <- as.factor(transf_glm_preds)

# Classification Matrix
conf_matrix_transf <- confusionMatrix(transf_glm_preds, transf_validation_outcome$class, positive = "1")
conf_matrix_transf

# glm model evaluation on Validation data
transf_glm_evaluation <- data.frame(conf_matrix_transf$byClass)
transf_glm_evaluation

#            Reference
# Prediction    1    2
#          1  231   62
#          2  807 7899

# False positive rate
62/(62+7899)

# AUC/ROC

# ROC Train
fit_transf <- fitted(glm_transf)
roc_transf_train <- roc(train_transf_rfe$class, fit_transf)
ggroc(roc_transf_train)
auc(roc_transf_train)
# Area under the curve: 0.9557

# ROC Test
roc_transf_test <- roc(transf_validation_outcome$class, transf_glm_probs)
ggroc(list(train=roc_transf_train, test=roc_transf_test), legacy.axes = TRUE) +
  ggtitle("ROC of Logistic Regression with Transformed RFE features and Highly Correlated features removed") +
  labs(color = "")
auc(roc_transf_test)
# Area under the curve: 0.8333

