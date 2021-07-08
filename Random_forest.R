# 05.23.2021

### Random Forest ###

# Load libraries
library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(randomForest)
library(caret)
library(pROC)

### Baseline Model ###

# Using All Local Features #

set.seed(2021)

# train / validation
train_lf <- train_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

## use df_lf with removed highly correlated features (from Feature-engineering.R)
# train_lf <- df_lf

validation_lf <-
  valid_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

# relevel to two factor levels instead of three
train_lf$class <- factor(train_lf$class, levels = c(1,2))
validation_lf$class <- factor(validation_lf$class, levels = c(1,2))

validation_lf_features <- validation_lf %>% select(-class) # features

# fit the model
rand_forest_lf <- randomForest(class~., data = train_lf, mtry = 9, importance =TRUE)
rand_forest_lf

# variable importances for an object created by randomForest
rf_var_imp_lf <- data.frame(importance(rand_forest_lf))
varImpPlot(rand_forest_lf)

preds_rand_forest_lf <- predict(rand_forest_lf, newdata = validation_lf_features)

# Classification Matrix
conf_matrix_lf <- confusionMatrix(validation_lf$class, preds_rand_forest_lf, positive = "1")
conf_matrix_lf

lf_rf_evaluation <- data.frame(conf_matrix_lf$byClass)
lf_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  912  126
#             2   24 7937

# false positive rate
126 / (126+7937)

# ROC Train
preds_rand_forest_lf_roc <- predict(rand_forest_lf, 
                                    newdata = validation_lf_features, 
                                    type="prob")
roc_rf_lf_train <- roc(train_lf$class, rand_forest_lf$votes[,2])
ggroc(roc_rf_lf_train)
auc(roc_rf_lf_train)
# 0.9975, 0.9978

# ROC Test
roc_rf_lf_test <- roc(validation_lf$class, preds_rand_forest_lf_roc[,2])
ggroc(list(train=roc_rf_lf_train, test=roc_rf_lf_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with All Local features") +
  labs(color = "")
auc(roc_rf_lf_test)
# 0.9888, 0.9885



# Using RFE Features #

set.seed(2021)

## use df_rfe with removed highly correlated features (from Feature-engineering.R)

# load RFE train (10 variables)
train_rfe <- df_rfe %>%
  filter(class != 3)
# relevel to two factor levels instead of three
train_rfe$class <- factor(train_lf$class, levels = c(1,2))

# use same validation
dim(validation_lf)
dim(validation_lf_features)

# fit the model
rand_forest_rfe <- randomForest(class~., data = train_rfe, mtry = 3, importance =TRUE)
# mtry = sqrt(ncol(train_rfe)-1) = √p = 3 (rounded down)
rand_forest_rfe

# variable importances for an object created by randomForest
rf_var_imp_rfe <- data.frame(importance(rand_forest_rfe))
varImpPlot(rand_forest_rfe)

preds_rand_forest_rfe <- predict(rand_forest_rfe, newdata = validation_lf_features)

# Classification Matrix
conf_matrix_rfe <- confusionMatrix(validation_lf$class, preds_rand_forest_rfe, positive = "1")
conf_matrix_rfe

rfe_rf_evaluation <- data.frame(conf_matrix_rfe$byClass)
rfe_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  793  245
#             2   28 7933

# false positive rate
245 / (245+7933)

# ROC Train
preds_rand_forest_rfe_roc <- predict(rand_forest_rfe, 
                                    newdata = validation_lf_features, 
                                    type="prob")
roc_rf_rfe_train <- roc(train_rfe$class, rand_forest_rfe$votes[,2])
ggroc(roc_rf_rfe_train)
auc(roc_rf_rfe_train)
# 0.9972

# ROC Test
roc_rf_rfe_test <- roc(validation_lf$class, preds_rand_forest_rfe_roc[,2])
ggroc(list(train=roc_rf_rfe_train, test=roc_rf_rfe_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with RFE features") +
  labs(color = "")
auc(roc_rf_rfe_test)
# 0.9823



# Using LVQ Features #

set.seed(2021)

## use df_lvq with removed highly correlated features (from Feature-engineering.R)

# load LVQ train (9 variables)
train_lvq <- df_lvq %>%
  filter(class != 3)
# relevel to two factor levels instead of three
train_lvq$class <- factor(train_lvq$class, levels = c(1,2))

# use same validation
dim(validation_lf)
dim(validation_lf_features)

# fit the model
rand_forest_lvq <- randomForest(class~., data = train_lvq, mtry = 3, importance =TRUE)
# mtry = sqrt(ncol(train_lvq)-1) = √p = 3 (rounded down)
rand_forest_lvq

# variable importances for an object created by randomForest
rf_var_imp_lvq <- data.frame(importance(rand_forest_lvq))
varImpPlot(rand_forest_lvq)

preds_rand_forest_lvq <- predict(rand_forest_lvq, newdata = validation_lf_features)

# Classification Matrix
conf_matrix_lvq <- confusionMatrix(validation_lf$class, preds_rand_forest_lvq, positive = "1")
conf_matrix_lvq

lvq_rf_evaluation <- data.frame(conf_matrix_lvq$byClass)
lvq_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  936  102
#             2   87 7874

# false positive rate
102 / (102+7874)

# ROC Train
preds_rand_forest_lvq_roc <- predict(rand_forest_lvq, 
                                     newdata = validation_lf_features, 
                                     type="prob")
roc_rf_lvq_train <- roc(train_lvq$class, rand_forest_lvq$votes[,2])
ggroc(roc_rf_lvq_train)
auc(roc_rf_lvq_train)
# 0.9916

# ROC Test
roc_rf_lvq_test <- roc(validation_lf$class, preds_rand_forest_lvq_roc[,2])
ggroc(list(train=roc_rf_lvq_train, test=roc_rf_lvq_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with LVQ features") +
  labs(color = "")
auc(roc_rf_lvq_test)
# 0.9861



# Using Autoencoder Features #

set.seed(2021)

# load AE train
ae_train <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_train.csv")
ae_train$class<- as.factor(ae_train$class)

# load AE validation
ae_validation <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_valid.csv")
ae_validation$class<- as.factor(ae_validation$class)

ae_validation_features <- ae_validation %>% select(-class) # predictor variables

# fit the model
rand_forest_ae <- randomForest(class~., data = ae_train, mtry = 4, importance =TRUE)
# mtry = sqrt(ncol(ae_train)-1) = √p = 4 (rounded down)
rand_forest_ae

# variable importances for an object created by randomForest
rf_var_imp_ae <- data.frame(importance(rand_forest_ae))
varImpPlot(rand_forest_ae)

preds_rand_forest_ae <- predict(rand_forest_ae, newdata = ae_validation_features)

# Classification Matrix
conf_matrix_ae <- confusionMatrix(ae_validation$class, preds_rand_forest_ae, positive = "1")
conf_matrix_ae

ae_rf_evaluation <- data.frame(conf_matrix_ae$byClass)
ae_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  793  245
#             2   67 7894

# false positive rate
245 / (245+7894)

# ROC Train
preds_rand_forest_ae_roc <- predict(rand_forest_ae, 
                                    newdata = ae_validation_features, 
                                    type="prob")
roc_rf_ae_train <- roc(ae_train$class, rand_forest_ae$votes[,2])
ggroc(roc_rf_ae_train)
auc(roc_rf_ae_train)
# 0.9851

# ROC Test
roc_rf_ae_test <- roc(ae_validation$class, preds_rand_forest_ae_roc[,2])
ggroc(list(train=roc_rf_ae_train, test=roc_rf_ae_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with Autoencoder features") +
  labs(color = "")
auc(roc_rf_ae_test)
# 0.9757



## Random Forest on Transformed data + DS (all Local Features) ##
            # and/or #
## Random Forest on Transformed data + DS + CORRELATION ##

# Data Preprocessing

set.seed(2021)

# all Local transformed features
# DO NOT RUN for correlated features removed, see below!
down_train <- down_train_lf # directly from "Transformation.R"
transf_valid <- valid_lf_trans # directly from "Transformation.R"

# transformed Local features DS + COR
# DO NOT RUN if all Local features needed, see above!
down_train <- df_lf_transf # directly from "Feature_engineering.R"
transf_valid <- df_transf_valid_lf_short # directly from "Feature_engineering.R"

validation_lf_features <- transf_valid %>% select(-class) # features

# fit the model
rand_forest_lf_tr <- randomForest(Class~., data = down_train, mtry = 6, importance =TRUE)
# mtry = √p = 9 (rounded down) - all Local Features
# mtry = √p = 6 (rounded down) - 40 Local Features
rand_forest_lf_tr

# variable importances for an object created by randomForest
rf_var_imp_lf_tr <- data.frame(importance(rand_forest_lf_tr))
varImpPlot(rand_forest_lf_tr, 
           main = "Variable Importance plot - Transformed Local features with Highly correlated features removed")

preds_rand_forest_lf_tr <- predict(rand_forest_lf_tr, newdata = validation_lf_features)

# Classification Matrix
conf_matrix_lf_tr <- confusionMatrix(transf_valid$class, preds_rand_forest_lf_tr, positive = "1")
conf_matrix_lf_tr

lf_tr_rf_evaluation <- data.frame(conf_matrix_lf_tr$byClass)
lf_tr_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  729  309
#             2  172 7789

#              Reference
#   Prediction    1    2
#            1  676  362
#            2  175 7786

# false positive rate
309 / (309+7789) # all local transf. features
362 / (362+7786) # LF with highly correlated features removed

# ROC Train
preds_rand_forest_lf_tr_roc <- predict(rand_forest_lf_tr, 
                                    newdata = validation_lf_features, 
                                    type="prob")
roc_rf_lf_tr_train <- roc(down_train$Class, rand_forest_lf_tr$votes[,2])
ggroc(roc_rf_lf_tr_train)
auc(roc_rf_lf_tr_train)
# 0.997,  0.9973

# ROC Test
roc_rf_lf_tr_test <- roc(transf_valid$class, preds_rand_forest_lf_tr_roc[,1])
ggroc(list(train=roc_rf_lf_tr_train, test=roc_rf_lf_tr_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with Transformed Local features and Highly Correlated features removed") +
  labs(color = "")
auc(roc_rf_lf_tr_test)
# 0.8592,  0.8614


## Random Forest on Transformed data + DS + RFE features ##

# Data Preprocessing

## RFE 20 features
rfe_features <- c("Class", "Local_2",  "Local_55", "Local_49", "Local_8", "Local_58", "Local_90", "Local_67", "Local_31",
                  "Local_3", "Local_16", "Local_73", "Local_52", "Local_28", "Local_18", "Local_4", "Local_40",
                  "Local_19", "Local_85", "Local_79", "Local_80")

# subset with transformed down-sampled data
rfe_train <- down_train_lf[, rfe_features]

# load transformed validation data
rfe_validation_features <- valid_lf_trans[, rfe_features[2:21]] # predictor variables

set.seed(2021)

# fit the model
rand_forest_rfe <- randomForest(Class~., data = rfe_train, mtry = 4, importance =TRUE)
# mtry = sqrt(ncol(rfe_train)-1) = √p = 4 (rounded down)
rand_forest_rfe

# variable importances for an object created by randomForest
rf_var_imp_rfe <- data.frame(importance(rand_forest_rfe))
varImpPlot(rand_forest_rfe, 
           main = "Variable Importance plot - Transformed Local RFE features and Highly correlated features removed")

preds_rand_forest_rfe <- predict(rand_forest_rfe, newdata = rfe_validation_features)

# Classification Matrix
conf_matrix_rfe <- confusionMatrix(valid_lf_trans$class, preds_rand_forest_rfe, positive = "1")
conf_matrix_rfe

rfe_rf_evaluation <- data.frame(conf_matrix_rfe$byClass)
rfe_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  689  349
#             2  199 7762

# false positive rate
349 / (349+7762)

# ROC Train
preds_rand_forest_rfe_roc <- predict(rand_forest_rfe, 
                                     newdata = rfe_validation_features, 
                                     type="prob")
roc_rf_rfe_train <- roc(rfe_train$Class, rand_forest_rfe$votes[,2])
ggroc(roc_rf_rfe_train)
auc(roc_rf_rfe_train)
# 0.9976

# ROC Test
roc_rf_rfe_test <- roc(valid_lf_trans$class, preds_rand_forest_rfe_roc[,2])
ggroc(list(train=roc_rf_rfe_train, test=roc_rf_rfe_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with Transformed Local RFE features and Highly correlated features removed") +
  labs(color = "")
auc(roc_rf_rfe_test)
# 0.8543


## Random Forest on Transformed data + DS + LVQ features ##

# Data Preprocessing

# for LVQ top 20 features.
lvq_features <- c("Class", "Local_55", "Local_90", "Local_49", "Local_31", "Local_18", "Local_91",
                  "Local_4", "Local_78", "Local_76", "Local_52", "Local_58", "Local_85",
                  "Local_40", "Local_73", "Local_37", "Local_16", "Local_8", "Local_92",
                  "Local_80", "Local_67")

lvq_train <- down_train_lf[, lvq_features]

# transform Validation data into the same shape as train data
lvq_validation_features <- valid_lf_trans[, lvq_features[-1]] # predictor variables

set.seed(2021)

# fit the model
rand_forest_lvq <- randomForest(Class~., data = lvq_train, mtry = 4, importance =TRUE)
# mtry = sqrt(ncol(lvq_train)-1) = √p = 4 (rounded down)
rand_forest_lvq

# variable importances for an object created by randomForest
rf_var_imp_lvq <- data.frame(importance(rand_forest_lvq))
varImpPlot(rand_forest_lvq, 
           main = "Variable Importance plot - Transformed Local LVQ features and Highly correlated features removed")

preds_rand_forest_lvq <- predict(rand_forest_lvq, newdata = lvq_validation_features)

# Classification Matrix
conf_matrix_lvq <- confusionMatrix(valid_lf_trans$class, preds_rand_forest_lvq, positive = "1")
conf_matrix_lvq

lvq_rf_evaluation <- data.frame(conf_matrix_lvq$byClass)
lvq_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  813  225
#             2  321 7640

# false positive rate
225 / (225+7640)

# ROC Train
preds_rand_forest_lvq_roc <- predict(rand_forest_lvq, 
                                     newdata = lvq_validation_features, 
                                     type="prob")
roc_rf_lvq_train <- roc(lvq_train$Class, rand_forest_lvq$votes[,2])
ggroc(roc_rf_lvq_train)
auc(roc_rf_lvq_train)
# 0.9958

# ROC Test
roc_rf_lvq_test <- roc(valid_lf_trans$class, preds_rand_forest_lvq_roc[,2])
ggroc(list(train=roc_rf_lvq_train, test=roc_rf_lvq_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with Transformed Local LVQ features and Highly correlated features removed") +
  labs(color = "")
auc(roc_rf_lvq_test)
# 0.8745



# DS + Autoencoder #

set.seed(2021)

# load Down-Sampled AE train data
ae_train <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_down_train.csv")
# or from "Logistic_reg.R":
# ae_train <- ae_train

# ae_train$class<- as.factor(ae_train$class)

# load AE validation
ae_validation <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_valid_new.csv")
# or from "Logistic_reg.R":
# ae_validation <- ae_validation
# ae_validation$class<- as.factor(ae_validation$class)

ae_validation_features <- ae_validation %>% select(-class) # predictor variables

# fit the model
rand_forest_ae <- randomForest(class~., data = ae_train, mtry = 4, importance =TRUE)
# mtry = sqrt(ncol(ae_train)-1) = √p = 4 (rounded down)
rand_forest_ae

# variable importances for an object created by randomForest
rf_var_imp_ae <- data.frame(importance(rand_forest_ae))
varImpPlot(rand_forest_ae, 
           main = "Variable Importance plot - Down-sampled Autoencoder features")

preds_rand_forest_ae <- predict(rand_forest_ae, newdata = ae_validation_features)

# Classification Matrix
conf_matrix_ae <- confusionMatrix(ae_validation$class, preds_rand_forest_ae, positive = "1")
conf_matrix_ae

ae_rf_evaluation <- data.frame(conf_matrix_ae$byClass)
ae_rf_evaluation

#               Reference
#    Prediction    1    2
#             1  938  100
#             2  496 7465 

# false positive rate
100 / (100+7465)

# ROC Train
preds_rand_forest_ae_roc <- predict(rand_forest_ae, 
                                    newdata = ae_validation_features, 
                                    type="prob")
roc_rf_ae_train <- roc(ae_train$class, rand_forest_ae$votes[,2])
ggroc(roc_rf_ae_train)
auc(roc_rf_ae_train)
# 0.9863

# ROC Test
roc_rf_ae_test <- roc(ae_validation$class, preds_rand_forest_ae_roc[,2])
ggroc(list(train=roc_rf_ae_train, test=roc_rf_ae_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with with Down-sampled Autoencoder features") +
  labs(color = "")
auc(roc_rf_ae_test)
# 0.9717



## Random Forest on Transformed data (all Local Features) ##
## Random Forest on Transformed data (all Local Features) + Upsample ##

# Data Preprocessing
set.seed(2021)

# all Local transformed features
transf_train_lf <- train_lf_trans # directly from "Transformation.R"
transf_valid_lf <- valid_lf_trans # directly from "Transformation.R"

validation_lf_features <- transf_valid_lf %>% select(-class) # features

# Up-Sampled data - all Local transformed features
transf_train_lf <- up_train_lf # directly from "Transformation.R"

# fit the model
rand_forest_lf_tr <- randomForest(Class~., data = transf_train_lf, mtry = 9, importance =TRUE)
# mtry = √p = 9 (rounded down) - all Local Features
rand_forest_lf_tr

# variable importances for an object created by randomForest
rf_var_imp_lf_tr <- data.frame(importance(rand_forest_lf_tr))
varImpPlot(rand_forest_lf_tr, 
           main = "Variable Importance plot - Up-sampled Transformed Local features")

preds_rand_forest_lf_tr <- predict(rand_forest_lf_tr, newdata = validation_lf_features)

# Classification Matrix
conf_matrix_lf_tr <- confusionMatrix(transf_valid_lf$class, preds_rand_forest_lf_tr, positive = "1")
conf_matrix_lf_tr

lf_tr_rf_evaluation <- data.frame(conf_matrix_lf_tr$byClass)
lf_tr_rf_evaluation

#               Reference     #               Reference
#    Prediction    1    2     #    Prediction    1    2
#             1  657  381     #             1  495  543
#             2   14 7947     #             2   13 7948

# false positive rate
381 / (381+7947) # all local transf. features
543 / (543+7948) # all local transf. features with Up-Sampling

# ROC Train
preds_rand_forest_lf_tr_roc <- predict(rand_forest_lf_tr, 
                                       newdata = validation_lf_features, 
                                       type="prob")
roc_rf_lf_tr_train <- roc(transf_train_lf$Class, rand_forest_lf_tr$votes[,2])
ggroc(roc_rf_lf_tr_train)
auc(roc_rf_lf_tr_train)
# 0.9972, 1

# ROC Test
roc_rf_lf_tr_test <- roc(transf_valid_lf$class, preds_rand_forest_lf_tr_roc[,1])
ggroc(list(train=roc_rf_lf_tr_train, test=roc_rf_lf_tr_test), legacy.axes = TRUE) +
  ggtitle("ROC of Random Forest with Up-Sampled Transformed Local features") +
  labs(color = "")
auc(roc_rf_lf_tr_test)
# 0.9016, 0.9022





