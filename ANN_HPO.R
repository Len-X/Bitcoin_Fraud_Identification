# HPO for ANN

# install.packages("tfruns")

# Load libraries
library(tidyverse)
library(ggplot2)
suppressPackageStartupMessages(library(keras))
library(tensorflow)
library(tfruns)
library(pROC)

# HPO of ANN on Local Features

# Data Preprocessing
# load data from "Transformation.R"

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
  flag_numeric('dropout1', 0.5),
  flag_numeric('dropout2', 0.5),
  flag_numeric('dropout3', 0.5),
  flag_integer('neurons1', 128),
  flag_integer('neurons2', 128),
  flag_integer('neurons3', 128),
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
  verbose = 1,
  callbacks = list(early_stop))

print(history)
plot(history)

score <- model %>% evaluate(x_valid, y_valid, 
                            batch_size = Batch_Size)
print(score)

# model tuning

hpo_flags <- list(
  dropout1 = c(0.4,0.5,0.6),
  dropout2 = c(0.3,0.4, 0.5),
  dropout3 = c(0.1, 0.2, 0.3),
  neurons1 = c(64,128,256),
  neurons2 = c(32,64,128),
  neurons3 = c(8,32,64),
  lr = c(0.0001,0.001,0.01))

hpo_runs <- tuning_run("Bitcoin_Fraud_Identification/ANN_HPO.R", sample = 0.1, 
                       flags = hpo_flags)

all_runs <- ls_runs(order = metric_val_accuracy, 
                    decreasing= TRUE, 
                    runs_dir = 'runs')

best_run <- all_runs[1,]; best_run
latest_run()

# save to csv
# write.csv(all_runs,"~/Desktop/Master/ann_hpo_runs.csv", 
#           row.names = FALSE)



