# Autoencoder

# install necessary packages
# install.packages("devtools")

# Load libraries
library(tidyverse)
library(ggplot2)
suppressPackageStartupMessages(library(keras))
library(deepviz) # model visualization

# Autoencoder on Local features (LF)

# set training data as matrix
x_train_local <- train_local %>% 
  filter(class != 3) %>% 
  select(4:96) %>%
  as.matrix()
# x_train_local <- as.matrix(x_train_local)

# relevel to two factor levels instead of three
df_train_local <- train_local %>% 
  filter(class != 3)
df_train_local$class <- factor(df_train_local$class, levels = c(1,2))

# set validation data as matrix
x_valid_local <- valid_local %>% 
  filter(class != 3) %>% 
  select(4:96) %>%
  as.matrix()

# relevel to two factor levels instead of three
df_valid_local <- valid_local %>% 
  filter(class != 3)
df_valid_local$class <- factor(df_valid_local$class, levels = c(1,2))

y_train_local <- df_train_local$class
y_valid_local <- df_valid_local$class


# AE with Down-Sapmpled data
# from "Transformation.R"

# set training data as matrix
x_train_local <- train_lf_down %>% 
  select(-Class) %>%
  as.matrix()

# set validation data as matrix
x_valid_local <- valid_lf %>% 
  select(-class) %>%
  as.matrix()

# set test data as matrix
x_test_local <- test_lf %>% 
  select(-class) %>%
  as.matrix()

# outcome variable
y_train_local <- train_lf_down$Class
y_valid_local <- valid_lf$class
y_test_local <- test_lf$class


set.seed(2021)

# build the model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 60, activation = "tanh", input_shape = ncol(x_train_local)) %>%
  layer_dense(units = 20, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 60, activation = "tanh") %>%
  layer_dense(units = ncol(x_train_local))

# view model layers
summary(model)
# plot model
model %>% plot_model()

# compile model
model %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

# fit model
model %>% fit(
  x = x_train_local, 
  y = x_train_local, 
  epochs = 500,  # try 100, 500, 1000
  validation_data = list(x_valid_local, x_valid_local),
  verbose = 1
)

# evaluate the performance of the model
mse_ae_train <- evaluate(model, x_train_local, x_train_local)
mse_ae_train

mse_ae_valid <- evaluate(model, x_valid_local, x_valid_local)
mse_ae_valid

mse_ae_test <- evaluate(model, x_test_local, x_test_local)
mse_ae_test

# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model$input, outputs = get_layer(model, "bottleneck")$output)
# train prediction
intermediate_output_train <- predict(intermediate_layer_model, x_train_local)
intermediate_output_train
# validation prediction
intermediate_output_valid <- predict(intermediate_layer_model, x_valid_local)
intermediate_output_valid
# test prediction
intermediate_output_test <- predict(intermediate_layer_model, x_test_local)
intermediate_output_test

# -------------------------------------------------------------------------------------------------------

# combine resulted AE features with class
# take an original train class
df_class1 <- train_local %>% 
  filter(class != 3) %>% 
  select(2)

df_class2 <- valid_local %>% 
  filter(class != 3) %>% 
  select(2)

# relevel to two factor levels instead of three
df_class1$class <- factor(df_class1$class, levels = c(1,2))
df_class2$class <- factor(df_class2$class, levels = c(1,2))

# combine class and AE-derived train features
df_ae_train <- cbind(df_class1, intermediate_output_train)
# combine class and AE-derived validation features
df_ae_valid <- cbind(df_class2, intermediate_output_valid)

# save to csv combined AE hidden layer output
write.csv(df_ae_train, "ae_20_variables_train.csv", row.names = FALSE)
write.csv(df_ae_valid, "ae_20_variables_valid.csv", row.names = FALSE)

# -------------------------------------------------------------------------------------------------------

# for down-sampled data

# combine class and AE-derived train features
df_ae_train <- cbind(y_train_local, intermediate_output_train)
# combine class and AE-derived validation features
df_ae_valid <- cbind(y_valid_local, intermediate_output_valid)
# combine class and AE-derived test features
df_ae_test <- cbind(y_test_local, intermediate_output_test)

# save to csv combined AE hidden layer output
write.csv(df_ae_train, "ae_20_down_train.csv", row.names = FALSE)
write.csv(df_ae_valid, "ae_20_variables_valid_new.csv", row.names = FALSE)
write.csv(df_ae_test, "ae_20_variables_test.csv", row.names = FALSE)


# Autoencoder on All Features (AF)

# Data Preprocessing

# load data from "Transformation.R"
x_train <- train_af %>%  # predictor variables
  select(-class) %>%
  as.matrix()

x_valid <- valid_af %>%  # predictor variables
  select(-class) %>%
  as.matrix()

x_test <- test_af %>%  # predictor variables
  select(-class) %>%
  as.matrix()

# outcome variable
y_train <- train_af$class
y_valid <- valid_af$class
y_test <- test_af$class

set.seed(2021)

# set AE model
model_ae <- keras_model_sequential()
model_ae %>%
  layer_dense(units = 60, activation = "tanh", input_shape = ncol(x_train)) %>%
  layer_dense(units = 20, activation = "tanh", name = "bottleneck") %>%
  layer_dense(units = 60, activation = "tanh") %>%
  layer_dense(units = ncol(x_train))

# view model layers
summary(model_ae)
# plot model
model_ae %>% plot_model()

# compile model
model_ae %>% compile(
  loss = "mean_squared_error", 
  optimizer = "adam"
)

# fit model
model_ae %>% fit(
  x = x_train, 
  y = x_train, 
  epochs = 100,  # try 100, 500, 1000
  validation_data = list(x_valid, x_valid),
  verbose = 1
)

# evaluate the performance of the model
mse_ae_train <- evaluate(model_ae, x_train, x_train)
mse_ae_train # 0.2546721

mse_ae_valid <- evaluate(model_ae, x_valid, x_valid)
mse_ae_valid # 1.056373

mse_ae_test <- evaluate(model_ae, x_test, x_test)
mse_ae_test # 0.9320067

# extract the bottleneck layer
intermediate_layer_model <- keras_model(inputs = model_ae$input, 
                                        outputs = get_layer(model_ae, "bottleneck")$output)
# train prediction
intermediate_output_train <- predict(intermediate_layer_model, x_train)
intermediate_output_train
# validation prediction
intermediate_output_valid <- predict(intermediate_layer_model, x_valid)
intermediate_output_valid
# test prediction
intermediate_output_test <- predict(intermediate_layer_model, x_test)
intermediate_output_test

# set anonymous variable names
col_prefix <- ("V")
n = 20
names_list <- unlist(mapply(function(x,y) 
  paste(x, seq(1,y), sep="_"), col_prefix, n))

# combine class and AE-derived train features
ae_train_af <- cbind(y_train, intermediate_output_train)
df_ae_train_af <- ae_train_af %>%
  as_tibble() %>%
  setNames(c("class", names_list))

# combine class and AE-derived validation features
ae_valid_af <- cbind(y_valid, intermediate_output_valid)
df_ae_valid_af <- ae_valid_af %>%
  as_tibble() %>%
  setNames(c("class", names_list))

# combine class and AE-derived test features
ae_test_af <- cbind(y_test, intermediate_output_test)
df_ae_test_af <- ae_test_af %>%
  as_tibble() %>%
  setNames(c("class", names_list))

# save to csv combined AE hidden layer output
write.csv(df_ae_train_af, "ae_20_AF_train.csv", row.names = FALSE)
write.csv(df_ae_valid_af, "ae_20_AF_valid.csv", row.names = FALSE)
write.csv(df_ae_test_af, "ae_20_AF_test.csv", row.names = FALSE)

