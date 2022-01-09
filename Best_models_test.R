# Testing best performing models on the test set
# Logistic Regression and Random Forest

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(pROC)
library(randomForest)

### Logistic Regression with AF + LVQ + CORR + IF (7 features) ###

# data preprocessing
lvq_features <- c("Local_5", "Local_18", "Local_43", "Local_53", "Local_66",
                  "Local_90", "Aggregated_39")
# Train
lvq_train <- train_af[, c("class", lvq_features)]

# transform Validation data into the same shape as train data
lvq_validation_features <- valid_af[, lvq_features] # predictor variables
lvq_validation_outcome <- valid_af %>% select(class) # outcome

# transform Test data into the same shape as train data
lvq_test_features <- test_af[, lvq_features] # predictor variables
lvq_test_outcome <- test_af %>% select(class) # outcome

## fit the GLM model
set.seed(2021)
glm_lvq <- glm(class ~ ., data=lvq_train, family=binomial)
summary(glm_lvq)

# validation data probabilities
lvq_glm_probs <- predict(glm_lvq, newdata=lvq_validation_features, type="response")
# test data probabilities
lvq_glm_probs_test <- predict(glm_lvq, newdata=lvq_test_features, type="response")

# predictions validation data
lvq_glm_preds = rep(1, 8999) # creates a vector of 8,999 class "1" elements
lvq_glm_preds[lvq_glm_probs >.5 ] = 2 # transforms to class "2" all of the elements 
# for which the predicted probability of class 2 exceeds 0.5

# predictions test data
lvq_glm_preds_test = rep(1, 11184) # creates a vector of 11,184 class "1" elements
lvq_glm_preds_test[lvq_glm_probs_test >.5 ] = 2

# set levels for predictions validation data
lvq_glm_preds <- as.factor(lvq_glm_preds)
# set levels for predictions test data
lvq_glm_preds_test <- as.factor(lvq_glm_preds_test)

# Classification Matrix test data
conf_matrix_lvq <- confusionMatrix(lvq_glm_preds_test, lvq_test_outcome$class, positive = "1")
conf_matrix_lvq

# glm model evaluation on test data
lvq_glm_evaluation <- data.frame(conf_matrix_lvq$byClass)
lvq_glm_evaluation

# AUC/ROC
# ROC Train
fit_lvq <- fitted(glm_lvq)
roc_lvq_train <- roc(lvq_train$class, fit_lvq)
# ROC Validation
roc_lvq_validation <- roc(lvq_validation_outcome$class, lvq_glm_probs)
# ROC Test
roc_lvq_test <- roc(lvq_test_outcome$class, lvq_glm_probs_test)
ggroc(list(train=roc_lvq_train, validation=roc_lvq_validation, test=roc_lvq_test), legacy.axes = TRUE) +
  ggtitle("ROC of best Logistic Regression model with LVQ on all features, Highly Correlated and unimportant features removed") +
  labs(color = "")

# AUC
auc(roc_lvq_train)
auc(roc_lvq_validation)
auc(roc_lvq_test)


### Random Forest on AF (Local + Aggregated) - 165 variables ###

# all features (directly from "Transformation.R")
train_all <- train_af
valid_all <- valid_af
test_all <- test_af

# test_all <- down_test_af # down-sampled test data (hypothesis testing only)
# test_all <- up_test_af # up-sampled test data (hypothesis testing only)

# features
validation_af_features <- valid_all %>% select(-class)
test_af_features <- test_all %>% select(-Class) #!

# fit the model
set.seed(2021)
rand_forest_af <- randomForest(class~., data = train_all, mtry = 12, importance =TRUE)
# mtry = √p = √165 = 12 (rounded down) - all features
rand_forest_af

# predictions
preds_rf_af_validation <- predict(rand_forest_af, newdata = validation_af_features)
preds_rf_af_test <- predict(rand_forest_af, newdata = test_af_features)

# Classification Matrix
conf_matrix_af <- confusionMatrix(test_af$class, preds_rf_af_test, positive = "1") # down_test_af$Class
conf_matrix_af

rf_evaluation_af <- data.frame(conf_matrix_af$byClass)
rf_evaluation_af

# AUC/ROC
# get probabilities
preds_rf_af_validation <- predict(rand_forest_af, newdata = validation_af_features, type="prob")
preds_rf_af_test <- predict(rand_forest_af, newdata = test_af_features, type="prob")

# ROC Train
roc_rf_af_train <- roc(train_all$class, rand_forest_af$votes[,2])
# ROC Validation
roc_rf_af_valid <- roc(valid_all$class, preds_rf_af_validation[,1])
# ROC Test
roc_rf_af_test <- roc(test_all$class, preds_rf_af_test[,1])
ggroc(list(train=roc_rf_af_train, validation=roc_rf_af_valid, test=roc_rf_af_test), legacy.axes = TRUE) +
  ggtitle("ROC of best Random Forest model on all features") +
  labs(color = "")

# AUC
auc(roc_rf_af_train)
auc(roc_rf_af_valid)
auc(roc_rf_af_test)




