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

# set positive class 1 as fraud amd 0 as licit (non-fraud)
levels(train_lf$class) <- c(1, 0)
levels(valid_lf$class) <- c(1, 0)
# alternative way
# levels(train_lf$class)[levels(train_lf$class)==2] <- 0

table(train_lf$class)
table(valid_lf$class)

# y_train <- as.numeric(train_lf$class)
# y_valid <- as.numeric(valid_lf$class)

y_train <- to_categorical(train_lf$class)
y_valid <- to_categorical(valid_lf$class)

# y_train <- train_lf$class %>% as.matrix() # outcome/target variable
# y_valid <- valid_lf$class %>% as.matrix() # outcome/target variable

# set "dimnames" to "NULL"
dimnames(x_train) <- NULL
dimnames(x_valid) <- NULL

## Build the NN model ##

set.seed(2021)

# initialize a sequential model
model <- keras_model_sequential()

# set model
model %>%
  layer_dense(units = 128, activation = "relu", input_shape = ncol(x_train)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 2, activation = "softmax")

# input_shape - number of input variables
# units = 2 - indicates 2 classes

# summary of a model
summary(model)

# model configuration
get_config(model)

# layer configuration
get_layer(model, index = 1)

# used to retrieve a flattened list of the model’s layers
model$layers

# list the input tensors
model$inputs

# list the output tensors
model$outputs

# compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = "accuracy")

# fit the model and store the fitting history
history <- model %>% fit(
  x_train, 
  y_train, 
  epochs = 100, 
  batch_size = 32, 
  validation_split = 0.2,
  verbose = 1)

# plot the history
plot(history)

# make predictions for the validation data
predictions <- model %>% predict_classes(x_valid, batch_size = 128)

# swap levels in predictions. Make 1 first
predictions <- relevel((as.factor(predictions)), "1")
table(predictions)

# Confusion matrix
table(valid_lf$class, predictions)

conf_matrix <- confusionMatrix(valid_lf$class, predictions)
conf_matrix

# evaluation by class
evaluation <- data.frame(conf_matrix$byClass)
evaluation

# evaluate the model
score <- model %>% evaluate(x_valid, y_valid, batch_size = 128)

# Print the score
print(score)


















