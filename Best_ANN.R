# Best ANN based on HPO

# Load libraries
library(tidyverse)
library(ggplot2)
suppressPackageStartupMessages(library(keras))
library(tensorflow)
library(tfruns)
library(pROC)

# open ANN HPO runs
hpo_runs <- read.csv("ann_hpo_runs.csv")
# access best run
best_run <- hpo_runs[1,]; best_run

# Data Preprocessing
# load data from "Transformation.R"

# predictor/outcome variables split (LF)
x_train <- train_lf %>%  # predictor variables
  select(-class) %>%
  as.matrix()

x_valid <- valid_lf %>%  # predictor variables
  select(-class) %>%
  as.matrix()

x_test <- test_lf %>%  # predictor variables
  select(-class) %>%
  as.matrix()

# set positive class 1 as fraud and 0 as licit (non-fraud)
levels(train_lf$class) <- c(1, 0)
levels(valid_lf$class) <- c(1, 0)
levels(test_lf$class) <- c(1, 0)

table(train_lf$class)
table(valid_lf$class)
table(test_lf$class)

y_train <- to_categorical(train_lf$class)  # outcome/target variable
y_valid <- to_categorical(valid_lf$class)  
y_test <- to_categorical(test_lf$class)

#___________________________________________________________________

# predictor/outcome variables split (AF)
x_train <- train_af %>%  # predictor variables
  select(-class) %>%
  as.matrix()

x_valid <- valid_af %>%  # predictor variables
  select(-class) %>%
  as.matrix()

# set positive class 1 as fraud and 0 as licit (non-fraud)
levels(train_af$class) <- c(1, 0)
levels(valid_af$class) <- c(1, 0)

table(train_af$class)
table(valid_af$class)

y_train <- to_categorical(train_af$class)  # outcome/target variable
y_valid <- to_categorical(valid_af$class)  # outcome/target variable

#___________________________________________________________________

# LF with down-sampling
x_train <- train_lf_down %>%  # predictor variables
  select(-Class) %>%
  as.matrix()

# set positive class 1 as fraud and 0 as licit (non-fraud)
levels(train_lf_down$Class) <- c(1, 0)

y_train <- to_categorical(train_lf_down$Class)  # outcome/target variable
table(train_lf_down$Class)

#___________________________________________________________________

# LF with up-sampling
x_train <- train_lf_up %>%  # predictor variables
  select(-Class) %>%
  as.matrix()

# set positive class 1 as fraud and 0 as licit (non-fraud)
levels(train_lf_up$Class) <- c(1, 0)

y_train <- to_categorical(train_lf_up$Class)  # outcome/target variable
table(train_lf_up$Class)

#___________________________________________________________________

# set "dimnames" to "NULL"
dimnames(x_train) <- NULL
dimnames(x_valid) <- NULL

# build the model
set.seed(2021)

# manual flags
# FLAGS <- flags(
#   flag_numeric('dropout1', 0.4),
#   flag_numeric('dropout2', 0.2),
#   flag_numeric('dropout3', 0.1),
#   flag_integer('neurons1', 64),
#   flag_integer('neurons2', 32),
#   flag_integer('neurons3', 8),
#   flag_numeric('lr', 0.001))

# train the model with best run's parameters
FLAGS = list(
  dropout1 = best_run$flag_dropout1,
  dropout2 = best_run$flag_dropout2,
  dropout3 = best_run$flag_dropout3,
  neurons1 = best_run$flag_neurons1,
  neurons2 = best_run$flag_neurons2,
  neurons3 = best_run$flag_neurons3,
  lr = best_run$flag_lr)

build_model <- function() {
  
  model <- keras_model_sequential() 
  model %>% 
    layer_dense(units = FLAGS$neurons1, activation = "relu", 
                input_shape = ncol(x_train)) %>%
    layer_dropout(FLAGS$dropout1) %>%
    layer_dense(units = FLAGS$neurons2, activation = "relu") %>%
    layer_dropout(FLAGS$dropout2) %>%
    layer_dense(units = FLAGS$neurons3, activation = "relu") %>%
    layer_dropout(FLAGS$dropout3) %>%
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(lr = FLAGS$lr),
    metrics = list("accuracy"))
  model
}

model <- build_model()

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 5)
Epochs <- 20
Batch_Size <- 128

# Fit the model and store training stats
history <- model %>% fit(
  x_train,
  y_train,
  epochs = Epochs,
  batch_size = Batch_Size,
  validation_data = list(x_valid, y_valid),
  verbose = 1,
  callbacks = list(early_stop))

print(history)
plot(history)

score <- model %>% evaluate(x_valid, y_valid, 
                            batch_size = Batch_Size)
print(score)

# make predictions for the validation data
predictions <- model %>% predict_classes(x_valid, batch_size = 128)
probabilities <- model %>% predict_proba(x_valid) %>% as.data.frame()
probabilities_train <- model %>% predict_proba(x_train) %>% as.data.frame()

# swap levels in predictions. Make 1 first
predictions <- relevel((as.factor(predictions)), "1")
table(predictions)

# confusion matrix
conf_matrix <- confusionMatrix(valid_lf$class, predictions)
conf_matrix

# evaluation by class
evaluation <- data.frame(conf_matrix$byClass)
evaluation

# ROC Train/Validation
roc_train <- roc(train_lf$class, probabilities_train$V1)
roc_valid <- roc(valid_lf$class, probabilities$V1)
ggroc(list(Train = roc_train, Validation = roc_valid), legacy.axes = TRUE) +
  ggtitle("ROC of best ANN model with Local features") +
  labs(color = "")
auc(roc_train) # 0.988, 0.9965, 0.9937, 0.9833, 0.9929
auc(roc_valid) # 0.9812, 0.9668, 0.9624, 0.9770, 0.9837

# save and load model
save_model_hdf5(model, "model_HPO_Auto.h5")
model <- load_model_hdf5("model_HPO_Auto.h5")

# test data predictions
predictions_test <- model %>% predict_classes(x_test, batch_size = 128)
probabilities_test <- model %>% predict_proba(x_test) %>% as.data.frame()

# swap levels in predictions. Make 1 first
predictions_test <- relevel((as.factor(predictions_test)), "1")
table(predictions_test)

# confusion matrix
conf_matrix_test <- confusionMatrix(test_lf$class, predictions_test)
conf_matrix_test

# evaluation by class
evaluation_test <- data.frame(conf_matrix_test$byClass)
evaluation_test

# ROC Test
roc_test <- roc(test_lf$class, probabilities_test$V1)
ggroc(list(Train = roc_train, Validation = roc_valid, Test = roc_test), 
      legacy.axes = TRUE) +
  ggtitle("ROC of best ANN model with Local features") +
  labs(color = "")
auc(roc_test) # 0.8416




