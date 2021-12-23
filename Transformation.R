# Data Transformation and Sampling 05.29.2021

# install necessary packeges
# install.packages("bestNormalize")
# install.packages("ROSE")
# install.packages("DMwR")
library(tidyverse)
library(ggplot2)
library(caret)
library(bestNormalize)
library(ROSE)
# library(DMwR) # SMOTE sampling

### Data Preprocessing ###

# load full df
df <- read.csv("Bitcoin_Fraud_Identification/Data/Bitcoin_Full_df.csv")
attach(df)

# convert class "unknown" to 3
levels(df$class) <- sub("unknown", 3, levels(df$class))

# workaround
# df$class[df$class == "unknown"] <- 3
# df$class <- as.numeric(factor(df$class))
# df$class <- as.factor(df$class)

### Train / Validation / Test Split ###

set.seed(2021)

# full df with all features
train_full <- df %>% filter(TimeStep <= 29)
valid_full <- df %>% filter(TimeStep > 29 & TimeStep <= 39)
test_full <- df %>% filter(TimeStep > 39)

# df with local features
train_local <- train_full %>% select(1:96)
valid_local <- valid_full %>% select(1:96)
test_local <- test_full %>% select(1:96)

# remove class 3 and variables txId and TimeStep on Local Features
train_lf <- train_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

valid_lf <- valid_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

test_lf <- test_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

# remove class 3 and variables txId and TimeStep on All Features (Local+Aggregated)
train_af <- train_full %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

valid_af <- valid_full %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

test_af <- test_full %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

# relevel to two factor levels instead of three
train_lf$class <- factor(train_lf$class, levels = c(1,2))
valid_lf$class <- factor(valid_lf$class, levels = c(1,2))
test_lf$class <- factor(test_lf$class, levels = c(1,2))
# train_lf[,-1]

train_af$class <- factor(train_af$class, levels = c(1,2))
valid_af$class <- factor(valid_af$class, levels = c(1,2))
test_af$class <- factor(test_af$class, levels = c(1,2))

# transform negative values in data for Box-Cox method: 
# find the lowest (negative) value
min(train_lf[,-1])  # [1] -6.996606
min(valid_lf[,-1])  # [1] -6.996606
min(test_lf[,-1])  # [1] -6.996606
# shift the distribution by adding a scalar 7 to the df 
# to make all values in the df positive
shitfed_train <- train_lf[,-1] + 7
shifted_valid <- valid_lf[,-1] + 7
shifted_test <- test_lf[,-1] + 7
# check for smallest value to make sure there are no negative values in the df
min(shitfed_train)  # [1] 0.003394184
min(shifted_valid)  # [1] 0.003394184
min(shifted_test)  # [1] 0.003394184


### Box-Cox Transformation ###

library(MASS)

#estimate a Box–Cox transformation 
bc_preprocess <- preProcess(shitfed_train, method = "BoxCox")

#transform data
bc_train <- predict(bc_preprocess, shitfed_train)

# check manually on Local_1 feature
bc_result <- data.frame(bc_preprocess$bc$Local_1$lambda)
lambda <- (-2)
# formula: (x^lambda-1)/lambda
bc_transformed <- (shitfed_train$Local_1^lambda-1)/lambda
bc_train$Local_1 == bc_transformed

# orignal histogram
hist(shitfed_train$Local_1)
# box-cox transformed histogram
hist(bc_transformed)

# with MASS library
bc1 <- boxcox(shitfed_train$Local_15 ~ shitfed_train$Local_90, 
              lambda = seq(-5000, 3, by = 0.1), 
              optimize = TRUE, objective.name="Shapiro-Wilk")


## Box-Cox Transformation with bestNormalize ##

# check Box-Cox again
boxcox_obj_1 <- boxcox(shitfed_train$Local_1)
boxcox_obj_1
hist(boxcox_obj_1$x.t)

### Yeo-Johnson Transformation  ###

yeojohnson_obj_1 <- yeojohnson(shitfed_train$Local_1)
yeojohnson_obj_1
hist(yeojohnson_obj_1$x.t)

# orderNorm Transformation
orderNorm_obj_1 <- orderNorm(shitfed_train$Local_1)
orderNorm_obj_1
hist(orderNorm_obj_1$x.t)   ## BEST METHOD! ##

# Visualiza the results
par(mfrow = c(2,2))
MASS::truehist(shitfed_train$Local_1, main = "Original", nbins = 12)
MASS::truehist(boxcox_obj_1$x.t, main = "Box-Cox transformation", nbins = 12)
MASS::truehist(yeojohnson_obj_1$x.t, main = "Yeo-Johnson transformation", nbins = 12)
MASS::truehist(orderNorm_obj_1$x.t, main = "orderNorm transformation", nbins = 12)
par(mfrow = c(1,1))

# compare all bestNormalize methods on one feature Local_1
BN_obj <- bestNormalize(shitfed_train$Local_1)
BN_obj
plot(BN_obj, leg_loc = "topleft", main="Normalization methods comparison")

# run bestNormalize methods on all Train variables
BN_obj_train_lf <- lapply(shitfed_train, function(x) bestNormalize(x))
BN_obj_train_lf

# transform variables and save the result to df
# efficient method
transformed_train_lf <- data.frame(predict(BN_obj_train_lf))
hist(transformed_train_lf$Local_1)

# manual method
# transformed_train_lf <- c() 
# for(i in 1:93){
#   transformed_train_lf <- cbind(transformed_train_lf, BN_obj_train_lf[[i]]$x.t) }

# combine transformed features with true class
df_class_train <- train_lf %>%
  select(1)

transformed_train_lf <- cbind(df_class_train, transformed_train_lf)

# save to csv
# write.csv(transformed_train_lf,"~/Desktop/MASTERS/Bitcoin/transformed_train_local_features.csv", row.names = FALSE)



# run bestNormalize methods on all Validation variables
BN_obj_valid_lf <- lapply(shifted_valid, function(x) bestNormalize(x))
BN_obj_valid_lf

# transform variables and save the result to df
# efficient method
transformed_valid_lf <- data.frame(predict(BN_obj_valid_lf))

# combine transformed features with true class
df_class_valid <- valid_lf %>%
  select(1)

transformed_valid_lf <- cbind(df_class_valid, transformed_valid_lf)

# save to csv
# write.csv(transformed_valid_lf,"~/Desktop/MASTERS/Bitcoin/transformed_valid_local_features.csv", row.names = FALSE)


# run bestNormalize methods on all Test variables
BN_obj_test_lf <- lapply(shifted_test, function(x) bestNormalize(x))
BN_obj_test_lf

# transform variables and save the result to df
# efficient method
transformed_test_lf <- data.frame(predict(BN_obj_test_lf))

# combine transformed features with true class
df_class_test <- test_lf %>%
  select(1)

transformed_test_lf <- cbind(df_class_test, transformed_test_lf)

# save to csv
# write.csv(transformed_test_lf,"~/Desktop/MASTERS/Bitcoin/transformed_test_local_features.csv", row.names = FALSE)



### Sampling of the Imbalanced data ###


# Load transformed data
train_lf_trans <- read.csv("Bitcoin_Fraud_Identification/Data/transformed_train_local_features.csv")
valid_lf_trans <- read.csv("Bitcoin_Fraud_Identification/Data/transformed_valid_local_features.csv")

# Load Autoencoder features
train_ae <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_train.csv")
valid_ae <- read.csv("Bitcoin_Fraud_Identification/Data/ae_20_variables_valid.csv")

# set "class" as factor for transformed data
train_lf_trans$class <- factor(train_lf_trans$class, levels = c(1,2))
valid_lf_trans$class <- factor(valid_lf_trans$class, levels = c(1,2))
# set "class" as factor for Autoencoder data
train_ae$class <- factor(train_ae$class, levels = c(1,2))
valid_ae$class <- factor(valid_ae$class, levels = c(1,2))

# Downsapling #

set.seed(2021)

# transformed data
down_train_lf <- downSample(x = train_lf_trans[,-1], y = train_lf_trans$class)

table(down_train_lf$Class)

#    1    2 
# 2871 2871 

# test data (just to check hypothesis)
down_test_af <- downSample(x = test_af[,-1], y = test_af$class)

# Autoencoder data
down_train_ae <- downSample(x = train_ae[,-1], y = train_ae$class)

table(down_train_ae$Class)

#    1    2 
# 2871 2871

# downsample before running Autoencoder
train_lf_down <- downSample(x = train_lf[,-1], y = train_lf$class)

table(train_lf_down$Class)

#    1    2 
# 2871 2871


# Upsampling #

set.seed(2021)

# transformed data
up_train_lf <- upSample(x = train_lf_trans[,-1], y = train_lf_trans$class)

table(up_train_lf$Class)

#     1     2 
# 23510 23510 

# upsampling of test data (to check hypothesis)
up_test_af <- upSample(x = test_af[,-1], y = test_af$class)

# Upsample before running Autoencoder / Raw data
train_lf_up <- upSample(x = train_lf[,-1], y = train_lf$class)

table(train_lf_up$Class)

#     1     2 
# 23510 23510 


# Hybrid sampling methods
# add new synthetic data points to the minority class and downsample the majority class

# ROSE method on transformed data
train_lf_rose <- ROSE(class~., data=train_lf_trans)$data

table(train_lf_rose$class)

#     2     1 
# 13179 13202

# ROSE method on raw data
train_rose_lf <- ROSE(class~., data=train_lf)$data

table(train_rose_lf$class)

#     2     1 
# 13061 13320 


### Using weights for the imbalanced data ###

# checking current proportion of classes
prop.table(table(train_lf$class)) # train
prop.table(table(valid_lf$class)) # validation
prop.table(table(test_lf$class)) # test


