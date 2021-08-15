### Artificial Neural Network ###   06.20.2021

# Load libraries
library(tidyverse)
library(ggplot2)
suppressPackageStartupMessages(library(keras))
library(tensorflow)
library(pROC)


# Neural Network on Local Features

# Data Preprocessing

# load data from "Transformation.R"
str(train_lf) # train data, local features
str(valid_lf) # validation data, local features

# predictor/outcome variables split
x_train <- train_lf %>%  # predictor variables
  select(-class) %>%
  as.matrix()

x_valid <- valid_lf %>%  # predictor variables
  select(-class) %>%
  as.matrix()

# set positive class 1 as fraud and 0 as licit (non-fraud)
levels(train_lf$class) <- c(1, 0)
levels(valid_lf$class) <- c(1, 0)
# alternative way
# levels(train_lf$class)[levels(train_lf$class)==2] <- 0

table(train_lf$class)
table(valid_lf$class)

# y_train <- as.numeric(train_lf$class)
# y_valid <- as.numeric(valid_lf$class)

y_train <- to_categorical(train_lf$class)  # outcome/target variable
y_valid <- to_categorical(valid_lf$class)  # outcome/target variable

# y_train <- train_lf$class %>% as.matrix() # outcome/target variable
# y_valid <- valid_lf$class %>% as.matrix() # outcome/target variable

# set "dimnames" to "NULL"
dimnames(x_train) <- NULL
dimnames(x_valid) <- NULL

## Build the NN model ##

set.seed(2021)

# initialize a sequential model
model <- keras_model_sequential()   # baseline model

model_2 <- keras_model_sequential() # 2-nd model

# set model
model %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

# 2nd model with Dropout
model_2 %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 256, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 2, activation = "softmax")

# input_shape - number of input variables
# units = 2 - indicates 2 classes

# summary of a model
summary(model)
summary(model_2)

# model configuration
get_config(model)
get_config(model_2)

# layer configuration
get_layer(model, index = 1)
get_layer(model_2, index = 1)

# used to retrieve a flattened list of the model’s layers
model$layers
model_2$layers

# list the input tensors
model$inputs
model_2$inputs

# list the output tensors
model$outputs
model_2$outputs

# compile the baseline model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy")

# compile the 2-nd model
model_2 %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy")

# fit the baseline model and store the fitting history
history <- model %>% fit(
  x_train, 
  y_train, 
  epochs = 100, 
  batch_size = 128, # try 50, 128, 256, 512
  validation_data = list(x_valid, y_valid),
  verbose = 1)

# fit the 2-nd model and store the fitting history
history_2 <- model_2 %>% fit(
  x_train, 
  y_train, 
  epochs = 100, 
  batch_size = 128,
  validation_data = list(x_valid, y_valid),
  verbose = 1)

# plot the history
print(history)
plot(history)

print(history_2)
plot(history_2)

# make predictions for the validation data, baseline model
predictions <- model %>% predict_classes(x_valid, batch_size = 128)
probabilities <- model %>% predict_proba(x_valid) %>% as.data.frame()
probabilities_train <- model %>% predict_proba(x_train) %>% as.data.frame()

# make predictions for the validation data, 2-nd model
predictions_2 <- model_2 %>% predict_classes(x_valid, batch_size = 128)
probabilities_2 <- model_2 %>% predict_proba(x_valid) %>% as.data.frame()
probabilities_train_2 <- model_2 %>% predict_proba(x_train) %>% as.data.frame()

# swap levels in predictions. Make 1 first
predictions <- relevel((as.factor(predictions)), "1")
table(predictions)

predictions_2 <- relevel((as.factor(predictions_2)), "1")
table(predictions_2)

# Confusion matrix
table(valid_lf$class, predictions)     # baseline model
table(valid_lf$class, predictions_2)   # 2-nd model

conf_matrix <- confusionMatrix(valid_lf$class, predictions)
conf_matrix   # baseline model

conf_matrix_2 <- confusionMatrix(valid_lf$class, predictions_2)
conf_matrix_2   # 2-nd model

# evaluation by class
evaluation <- data.frame(conf_matrix$byClass)
evaluation  # baseline model

evaluation_2 <- data.frame(conf_matrix_2$byClass)
evaluation_2  # 2-nd model

# evaluate the model
score <- model %>% evaluate(x_valid, y_valid, batch_size = 128) # baseline model
score_2 <- model_2 %>% evaluate(x_valid, y_valid, batch_size = 128) # 2-nd model

# Print the score
print(score)    # baseline model
print(score_2)  # 2-nd model

# ROC Test for baseline model
roc_train <- roc(train_lf$class, probabilities_train$V1)
roc_test <- roc(valid_lf$class, probabilities$V1)
ggroc(list(Train = roc_train, Validation = roc_test), legacy.axes = TRUE) +
  ggtitle("ROC of Baseline ANN with Local features") +
  labs(color = "")
auc(roc_train) # 0.9991
auc(roc_test) # 0.937

# ROC Test for 2-nd model
roc_train_2 <- roc(train_lf$class, probabilities_train_2$V1)
roc_test_2 <- roc(valid_lf$class, probabilities_2$V1)
ggroc(list(Train = roc_train_2, Validation = roc_test_2), legacy.axes = TRUE) +
  ggtitle("ROC of 2nd ANN model with Local features") +
  labs(color = "")
auc(roc_train_2) # 0.9975
auc(roc_test_2) # 0.9846


# ANN 3rd model with L2-regularization

model_3 <- keras_model_sequential() # 3rd model

# set model with λ value = 0.001
model_3 %>%
  layer_dense(units = 256, activation = "relu", input_shape = ncol(x_train),
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 256, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 64, activation = "relu",
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 2, activation = "softmax")

summary(model_3)

# compile the 3rd model
model_3 %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy")

# fit the baseline model and store the fitting history
history_3 <- model_3 %>% fit(
  x_train, 
  y_train, 
  epochs = 100, 
  batch_size = 128,
  validation_data = list(x_valid, y_valid),
  verbose = 1)

print(history_3)
plot(history_3)

# make predictions for the validation data, 3rd model
predictions_3 <- model_3 %>% predict_classes(x_valid, batch_size = 128)
probabilities_3 <- model_3 %>% predict_proba(x_valid) %>% as.data.frame()
probabilities_train_3 <- model_3 %>% predict_proba(x_train) %>% as.data.frame()

# swap levels in predictions. Make 1 first
predictions_3 <- relevel((as.factor(predictions_3)), "1")
table(predictions_3)

# confusion matrix
conf_matrix_3 <- confusionMatrix(valid_lf$class, predictions_3)
conf_matrix_3

# evaluation by class
evaluation_3 <- data.frame(conf_matrix_3$byClass)
evaluation_3

score_3 <- model_3 %>% evaluate(x_valid, y_valid, batch_size = 128)
print(score_3)

# ROC Test for 3rd model
roc_train_3 <- roc(train_lf$class, probabilities_train_3$V1)
roc_test_3 <- roc(valid_lf$class, probabilities_3$V1)
ggroc(list(Train = roc_train_3, Validation = roc_test_3), legacy.axes = TRUE) +
  ggtitle("ROC of 3rd ANN model with Local features") +
  labs(color = "")
auc(roc_train_3) 
auc(roc_test_3) 



## Compare the models

# plot the model loss of the training data (baseline model)
plot(history$metrics$loss, main="Baseline Model Loss", 
     xlab = "epoch", 
     ylab="loss", 
     col="coral", 
     type="l",
     ylim = c(0,1), lwd = 2)
# plot the model loss of the test data
lines(history$metrics$val_loss, col="darkturquoise", lwd = 2)
# add legend
legend("topright", c("train","validation"), col=c("coral", "darkturquoise"), lty=c(1,1))

# plot the model accuracy of the training data (baseline model)
plot(history$metrics$acc, main="Baseline Model Accuracy", 
     xlab = "epoch", 
     ylab="loss", 
     col="coral", 
     type="l",
     ylim = c(0.6, 1), lwd = 2)
# model accuracy of the test data
lines(history$metrics$val_acc, col="darkturquoise", lwd = 2)
legend("bottomright", c("train","validation"), col=c("coral", "darkturquoise"), lty=c(1,1))











