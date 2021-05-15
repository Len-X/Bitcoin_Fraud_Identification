# Classification Models   05.15.2021

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(pROC)

### Baseline Model ###

### Logistic Regression ###

# Fit Logistic Regression to RFE data

# remove class 3 from the RFE df
rfe_train <-
  df_rfe %>% 
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

# fit the model

glm_rfe <- glm(class ~ ., data=rfe_train, family=binomial)

summary(glm_rfe)





