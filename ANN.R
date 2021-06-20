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

y_train <- train_lf$class  # outcome/target variable
y_valid <- valid_lf$class  # outcome/target variable

# set "dimnames" to "NULL"
dimnames(x_train) <- NULL
dimnames(x_valid) <- NULL


