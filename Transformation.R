# Data Transformation and Sampling 05.29.2021

# install necessary packeges
# install.packages("bestNormalize")
library(tidyverse)
library(ggplot2)


### Data Preprocessing ###


# load full df
df <- read.csv("Bitcoin_Fraud_Identification/Data/Bitcoin_Full_df.csv")
attach(df)

# convert class "unknown" to 3
levels(df$class) <- sub("unknown", 3, levels(df$class))

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

# remove class 3 and variables txId and TimeStep
train_lf <- train_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

valid_lf <- valid_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

test_lf <- test_local %>% 
  filter(class != 3) %>%
  select(-c(txId, TimeStep))

# relevel to two factor levels instead of three
train_lf$class <- factor(train_lf$class, levels = c(1,2))
valid_lf$class <- factor(valid_lf$class, levels = c(1,2))
test_lf$class <- factor(test_lf$class, levels = c(1,2))
# train_lf[,-1]

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
library(caret)
library(bestNormalize)

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


### Yeo-Johnson Transformation  ###

yeojohnson_obj_1 <- yeojohnson(shitfed_train$Local_1)
yeojohnson_obj_1
hist(yeojohnson_obj_1$x.t)

# check Box-Cox again
boxcox_obj_1 <- boxcox(shitfed_train$Local_1)
boxcox_obj_1
hist(boxcox_obj_1$x.t)

# orderNorm Transformation
orderNorm_obj_1 <- orderNorm(shitfed_train$Local_1)
orderNorm_obj_1
hist(orderNorm_obj_1$x.t)   ## BEST METHOD! ##

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

# save to csv
# write.csv(transformed_train_lf,"~/Desktop/MASTERS/Bitcoin/transformed_train_local_features.csv", row.names = FALSE)

# run bestNormalize methods on all Validation variables
BN_obj_valid_lf <- lapply(shitfed_valid, function(x) bestNormalize(x))
BN_obj_valid_lf

# transform variables and save the result to df
# efficient method
transformed_valid_lf <- data.frame(predict(BN_obj_valid_lf))





