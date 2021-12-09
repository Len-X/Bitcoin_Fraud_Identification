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

# set positive class 1 as fraud and 0 as licit (non-fraud)
levels(train_lf$class) <- c(1, 0)
levels(valid_lf$class) <- c(1, 0)

table(train_lf$class)
table(valid_lf$class)

y_train <- to_categorical(train_lf$class)  # outcome/target variable
y_valid <- to_categorical(valid_lf$class)  # outcome/target variable

# set "dimnames" to "NULL"
dimnames(x_train) <- NULL
dimnames(x_valid) <- NULL

# build the model
set.seed(2021)

FLAGS <- flags(
  flag_numeric('dropout1', 0.6),
  flag_numeric('dropout2', 0.3),
  flag_numeric('dropout3', 0.2),
  flag_integer('neurons1', 64),
  flag_integer('neurons2', 128),
  flag_integer('neurons3', 8),
  flag_numeric('lr', 0.001))

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
  class_weight = list("1"=4.6,"0"=0.56), # class weights
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
roc_test <- roc(valid_lf$class, probabilities$V1)
ggroc(list(Train = roc_train, Validation = roc_test), legacy.axes = TRUE) +
  ggtitle("ROC of best ANN model with class weight using Local features") +
  labs(color = "")
auc(roc_train) # 
auc(roc_test) # 
