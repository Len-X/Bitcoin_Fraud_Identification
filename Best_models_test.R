# Testing best performed models on the test set

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
auc(roc_lvq_train) # 0.943
auc(roc_lvq_validation) # 0.9045
auc(roc_lvq_test) # 0.8265



