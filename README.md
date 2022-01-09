# Identification of Fraudulent Transactions in Financial Data using Machine Learning Methods
## Master Thesis

The goal of our thesis research is to identify fraudulent Bitcoin transactions by developing and evaluating different Machine Learning methods. 

We have applied and experimented with three Machine Learning classification methods to identify illicit Bitcoin transactions: Logistic Regression, Random Forest and Artificial Neural Network (ANN). We took a holistic approach to this binary classification problem and ran a multitude of experiments with
each of these methods by using different inputs. 

We implemented and considered Spearman
correlation as a possible feature selection method. 

Additionally we performed a data transformation such that it follows normal distribution and studied its effect. 

We applied and examined RFE and LVQ feature selection methods to extract various subsets of features. 

We also used Autoencoder as a feature extraction method, and studied the effect of the encoded attributes on the performance of various models.

During the selection of the best performed ANN model, a hyperparameter optimization was conducted using both manual and automated processes.

And finally, we performed our experiments on various combinations of the aforementioned methods on the train and validation data, and selected three top performing models for testing on the test data. 

Out of three examined Machine Learning methods, two show the most promising results: Random Forest and Neural Network. Their best models performance outcomes are very close, with the Random Forest model yielding slightly superior results.

#### File Description

R script files:

- `EDA_bitcoin.R` - exploratory data analysis and normality tests.
- `Transformation.R` - includes data transformations and sampling methods.
- `Feature_engineering.R` - feature selection and correlation techniques.
- `Autoencoder.R` - contains Autoencoder models for feature extraction.
- `Logistic_reg.R` - development and evaluation of Logistic Regression models.
- `Random_forest.R` - development and evaluation of Random Forest models.
- `ANN.R` - development and evaluation of Artificial Neural Network models.
- `ANN_HPO.R` - hyperparameter optimization of Artificial Neural Network model.
- `Best_models_test.R` - testing best performing models on test data set.

Data folders: `elliptic_bitcoin_dataset` and `Data`:

`elliptic_bitcoin_dataset` - contains raw Bitcoin transactions data set via 3 csv files:

- [elliptic_txs_classes.csv](/elliptic_bitcoin_dataset/elliptic_txs_classes.csv) - contains tax ids and classes.
- [elliptic_txs_edgelist.csv](/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv) - contains edges linking two tax ids.
- [elliptic_txs_features.csv](/elliptic_bitcoin_dataset/elliptic_txs_features.csv) - contains tax ids, time steps and all local and aggregated features.

`Data` - contains processed data csv files:

- [Bitcoin_Full_df.csv](/Data/Bitcoin_Full_df.csv) - consolidated data set of Bitcoin tax ids, classes, and features.
- [ae_20_AF_test.csv](/Data/Bitcoin_Full_df.csv) - Autoencoder extracted features of all features of the test data set. 
- [ae_20_AF_train.csv](/Data/ae_20_AF_train.csv) - Autoencoder extracted features of all features of the train data set.
- [ae_20_AF_valid.csv](/Data/ae_20_AF_valid.csv) - Autoencoder extracted features of all features of the validation data set.
- [ae_20_down_train.csv](/Data/ae_20_down_train.csv) - Autoencoder extracted features of Local features with down-sampling of the train data set.
- [ae_20_variables_test.csv](/Data/ae_20_variables_test.csv) - Autoencoder extracted features of Local features of the test data set.
- [ae_20_variables_train.csv](/Data/ae_20_variables_train.csv) - Autoencoder extracted features of Local features of the train data set.
- [ae_20_variables_valid.csv](/Data/ae_20_variables_valid.csv) - Autoencoder extracted features of Local features of the validation data set.
- [transformed_test_local_features.csv](/Data/transformed_test_local_features.csv) - transformed Local features of the test data set.
- [transformed_train_local_features.csv](/Data/transformed_train_local_features.csv) - transformed Local features of the train data set.
- [transformed_valid_local_features.csv](/Data/transformed_valid_local_features.csv) - transformed Local features of the validation data set.



