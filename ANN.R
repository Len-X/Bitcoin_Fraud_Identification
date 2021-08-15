### Artificial Neural Network ###   06.20.2021

# Load libraries
library(tidyverse)
library(ggplot2)
suppressPackageStartupMessages(library(keras))
library(tensorflow)


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

model_2 %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 128, activation = "relu") %>%
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

# make predictions for the validation data, 2-nd model
predictions_2 <- model_2 %>% predict_classes(x_valid, batch_size = 128)
probabilities_2 <- model_2 %>% predict_proba(x_valid) %>% as.data.frame()

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
roc_test <- roc(valid_lf$class, probabilities$V1)
ggroc(list(Validation = roc_test), legacy.axes = TRUE) +
  ggtitle("ROC of ANN with Local features") +
  labs(color = "")
auc(roc_test)

# ROC Test for 2-nd model
roc_test_2 <- roc(valid_lf$class, probabilities_2$V1)
ggroc(list(Validation = roc_test_2), legacy.axes = TRUE) +
  ggtitle("ROC of ANN with Local features") +
  labs(color = "")
auc(roc_test_2)














