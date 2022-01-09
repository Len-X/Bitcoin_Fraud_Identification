# EDA of Bitcoin data

# Install necessary packeges
# Install packages if necessary
# install.packages("ggpubr")
# install.packages("hrbrthemes")
# install.packages("broom")
# install.packages("dgof")

# Load libraries
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(ggpubr)     # for an easy ggplot2-based data visualization
library(hrbrthemes)
library(broom)      # for function `tidy`
library(dgof)       # for Kolmogorov-Smirnov Test

# load data
df1 <- read.csv("Bitcoin_Fraud_Identification/elliptic_bitcoin_dataset/elliptic_txs_classes.csv")
df2 <- read.csv("Bitcoin_Fraud_Identification/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv")
df3 <- read.csv("Bitcoin_Fraud_Identification/elliptic_bitcoin_dataset/elliptic_txs_features.csv", header = FALSE)

# data shape
dim(df1)
dim(df2)
dim(df3)

glimpse(df1)
glimpse(df2)
glimpse(df3)

str(df1)
str(df2)
str(df3)

# count of each class
table(df1$class)
#    1       2 unknown 
# 4545   42019  157205 

# fraction of the Illicit transactions:
(4545 / (4545+42019)) * 100
# 9.760759%

# plot class frequencies
ggplot(df1 %>% 
         count(class) %>%             # Group by class, then count number in the group
         mutate(pct=n/sum(n)),        # Calculate percent within each class 
       aes(class, n, fill=class)) +
  geom_bar(stat="identity") +
  geom_text(aes(label=paste0(sprintf("%1.1f", pct*100),"%")), 
            position=position_stack(vjust=0.5)) +
  theme_bw()+
  ggtitle("Class Frequencies") +
  ylab("Number of Transactions") +
  xlab("Class") +
  scale_x_discrete(breaks=c("1", "2", "unknown"),
                   labels=c("Illicit", "Licit", "Unknown")) +
  theme(axis.text.x = element_text(size=11)) +
  theme(axis.text.y = element_text(size=11)) +
  theme(legend.position = "none")+
  theme(plot.title = element_text(hjust = 0.5))


# number of transactions in each time step
table(df3$V2)
plot(table(df3$V2))


# Merge two data frames df1(Classes) and df2(Features)

# rename first variable V1 and second variable V2 in df3 to txId and TimeStep respectively 
# to match the TaxId of df1
df3 <- rename(df3, txId = V1, TimeStep = V2)

# rename the rest of the variables
# define a list of varying "varname"
varname <- c('Local', 'Aggregated')
# define number of times above "varname" repeats itself
n <- c(93, 72) # 'Local' feature repeats 93 times and 'Aggregated' 72 times
# replace column names
names(df3)[3:ncol(df3)] <- unlist(mapply(function(x,y) paste(x, seq(1,y), sep="_"), varname, n))


# Merge two data frames by tax id while preserving the indices
full_df <- merge(df1, df3, by = "txId", sort=FALSE)

# Saving as csv file
# write_csv(full_df, "Bitcoin_Fraud_Identification/Data/Bitcoin_Full_df.csv")


# plot time frequencies by class
freq.time <- data.frame(table(full_df$TimeStep, full_df$class))

plot(freq.time$Var1, freq.time$Freq, xlab = "Time Step", ylab = "Transaction Frequency")
# lines(freq.time$Var1, freq.time$Freq, type = "l", lty = 1, xlab = "Time Step", ylab = "Frequency")
# title(xlab="Time Step", ylab="Transaction Frequency")


# transaction counts by TimeStep and class
trans_counts <- full_df %>%
  count(TimeStep, class)

# line plot
plot1 <- full_df %>%
  count(TimeStep, class) %>% 
  ggplot(mapping = aes(x = TimeStep, y = n, color = class)) +
  geom_line() +
  geom_point() +
  theme_bw()+
  ggtitle("Number of transactions at each time step") +
  ylab("Number of Transactions") +
  xlab("Time Step") +
  scale_x_continuous(breaks = seq(0, 50, by = 5)) +
  theme(axis.text.x = element_text(size=11)) +
  theme(axis.text.y = element_text(size=11)) +
  theme(legend.position = "none")+
  theme(plot.title = element_text(hjust = 0.5))

plot1

# bar plot
plot2 <- full_df %>% 
  count(TimeStep, class) %>%
  mutate(count=n()) %>%
  ggplot(aes(TimeStep, n, fill=class)) +
  geom_bar(stat="identity") +
  theme_bw()+
  ylab("Number of Transactions") +
  xlab("Time Step") +
  scale_x_continuous(breaks = seq(0, 50, by = 5)) +
  scale_fill_discrete(name = "Class", labels = c("Illicit", "Licit", "Unknown")) +
  theme(axis.text.x = element_text(size=11)) +
  theme(axis.text.y = element_text(size=11)) +
  theme(legend.position = c(0.95, 0.82))
  
plot2

# Arrange plots one on top of the other
p <- grid.arrange(plot1, plot2, nrow=2)
p

total_class <- full_df %>% 
  count(TimeStep, class) %>%
  add_count(TimeStep, wt = n, name = "total")

# line plot of all transactions by class & total
plot3 <- total_class %>%
  ggplot(mapping = aes(x = TimeStep, y = n, color = class)) +
  geom_line(aes(y = total, color = "total")) +
  geom_line() +
  geom_point() +
  geom_point(aes(y = total, color = "total")) +
  theme_bw()+
  ggtitle("Number of transactions at each time step") +
  ylab("Number of Transactions") +
  xlab("Time Step") +
  scale_x_continuous(breaks = seq(0, 50, by = 5)) +
  scale_color_manual(values = c(total = "#F8766D", "#00BA38", "darkgrey", "#619CFF"),
                     labels = c("Illicit", "Licit", "Total", "Unknown")) +
  theme(axis.text.x = element_text(size=11)) +
  theme(axis.text.y = element_text(size=11)) +
  theme(legend.position = "right") + # "none"
  theme(plot.title = element_text(hjust = 0.5))

plot3

# check for duplicate Tax Ids
anyDuplicated(full_df$txId) # no duplicate txId values

# check for NAs
table(is.na(full_df) == TRUE) # no missing values

# ___________________________________________________________


# It looks as all local and aggregated features have been normalized.
summary(full_df)                # all means ≈ 0

df_summary <- full_df %>%
  select(4:168) %>%
  apply(MARGIN = 2, FUN = summary)

# check mean and stansard deviation
sd(full_df$Aggregated_10)       # sd ≈ 1
var(full_df$Aggregated_10)      # variance ≈ 1

# function to get means and sd
stat_func <- function(x) {
  c(mean = mean(x), sd = sd(x))
}

df_stats <- full_df %>%
  select(4:168) %>%
  sapply(stat_func)

# check visually
boxplot(full_df$Aggregated_10)
boxplot(full_df$Local_27)


# Let us check if the data has been normalized.


# Attach full_df for easier use
attach(full_df)


# Assess the normality of the data.
# Visual method

# quick investigation of density
plot(density(full_df$Local_20)) 
plot(density(full_df$Local_67))
plot(density(full_df$Aggregated_12),xlim = c(-1,1))

range(full_df$Aggregated_71)

ggdensity(Local_10, 
          main = "Density plot",
          xlab = "Local_10 feature")
plot(density(Local_10))

ggqqplot(Local_10)

# As most of the points fall approximately along the reference line, we could assume normality.
# But some data points are way off, so this visual method is unreliable.

#  A Histogram of the variable might give an indication of the shape of the
# distribution. A density curve smoothes out the histogram and can be added to the graph.
hist(Local_10, probability=T, main="Histogram of Local_10",
     xlab="Local_10 distribution")
lines(density(Local_10),col=2)

# It is also difficult to derive the normality of the data from the histogram.


# Class density
plot4 <- ggplot(data=full_df, aes(x=TimeStep, group=class, fill=class)) +
  geom_density(adjust=1.5, alpha=.4) +
  theme_ipsum()
plot4

# Using small multiple
plot5 <- ggplot(data=full_df, aes(x=TimeStep, group=class, fill=class)) +
  geom_density(adjust=1.5, alpha=.4) +
  theme_ipsum() +
  facet_wrap(~class) +
  theme(
    legend.position="none",
    panel.spacing = unit(0.1, "lines"),
    axis.ticks.x=element_blank())
plot5


# It is difficult assess from the density plots since almost all result in L-shaped plots.


# The Shapiro-Wilk’s normality test

# Shapiro-Wilk’s method is widely recommended for normality test 
# and it provides better power than Kolmogorov-Smirnov (K-S). 
# It is based on the correlation between the data and the corresponding normal scores.

# The null hypothesis of these tests is that “sample distribution is normal” 
# (the population is distributed normally).
# If the test is significant, the distribution is non-normal.

# The R function shapiro.test() can be used to perform the Shapiro-Wilk test of 
# normality for one variable (univariate).
# For shapiro.test sample size must be between 3 and 5000
set.seed(2021)
var_sample <- sample_n(full_df, 5000)
var_sample <- var_sample[, 4:168]

shapiro.test(var_sample$Local_1)
# data:  var_sample$Local_1
# W = 0.20005, p-value < 2.2e-16

# From the output, the p-values is very small (p-value < 0.05) implying that the distribution 
# of the data are significantly different from normal distribution. 
# Don' have enough evidence to support the null hypothesis. We reject the null hypothesis.

# p = 2.2e-16 suggesting strong evidence of non-normality and a nonparametric test should be used. 

shapiro_test_results <- var_sample %>% 
  gather(key = "variable_name", value = "value") %>% 
  group_by(variable_name)  %>% 
  do(tidy(shapiro.test(.$value))) %>% 
  ungroup() %>% 
  select(-method)

shapiro_test_results

# p-value for every varible is extremely low. 
# We reject the assumption of the normality of the data.
# The distribution of the given data is significantly different from normal distribution.
# We reject the null hypothesis.


# Kolmogorov-Smirnov Test

# The Kolmogorov-Smirnov Test is a type of non-parametric test 
# to test normality of the sample.
# These tests provide a means of comparing distributions, whether two sample distributions 
# or a sample distribution with a theoretical distribution. The distributions are compared in 
# their cumulative form as empirical distribution functions. 
# The test statistic developed by Kolmogorov and Smirnov to compare distributions was simply
# the maximum vertical distance between the two functions.

# One-sample Kolmogorov-Smirnov test.
# AKA Kolmogorov-Smirnov goodness of fit test. 
# It assesses the degree of agreement between an observed distribution and a completely
# specified theoretical continuous distribution.

# When the sample size is larger than 100, the function ks.test will be numerically
# unstable
set.seed(2021)
var_sample_ks <- sample_n(full_df, 100)
var_sample_ks <- var_sample_ks[, 4:168]

# Let us use the same sample size as with Shapiro test
ks.test(var_sample_ks$Local_1, "pnorm", mean=0, sd=1) 
ks.test(var_sample_ks$Local_1, "pnorm", 0, 1)

ks_test_results <- sapply(var_sample_ks, function(y) {
  ks <- ks.test(var_sample_ks, y)
  c(statistic=ks$statistic, p.value=ks$p.value)
  setNames(c(ks$statistic, ks$p.value, ks$alternative, ks$method), 
           c("statistic", "p.value", "alternative", "method"))
})
ks_test_results

# We don't have enough evidence to support the null hypothesis H0. 
# All features do not come from the normal distribution.
# small p-values suggesting strong evidence of non-normality. We reject H0 hypothesis.


# detach full_df
detach(full_df)



