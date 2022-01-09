# Identification of Fraudulent Transactions in Financial Data using Machine Learning Methods
## Master Thesis

The goal of our thesis research is to identify fraudulent Bitcoin transactions by developing and evaluating different Machine Learning methods. We have applied and experimented with three Machine Learning classification methods to identify illicit Bitcoin transactions: Logistic Regression, Random Forest and Artificial Neural Network (ANN). We took a holistic approach to this binary classification problem and ran a multitude of experiments with
each of these methods by using different inputs. We implemented and considered Spearman
correlation as a possible feature selection method. Additionally we performed a data transformation such that it follows normal distribution and studied its effect. We applied and examined RFE and LVQ feature selection methods to extract various subsets of features. We also used Autoencoder as a feature extraction method, and studied the effect of the encoded attributes on the performance of various models. And finally, we performed our experiments on various combinations of the aforementioned methods on the validation data, and selected three top performing models. During the selection of the best performed ANN model, a hyperparameter optimization was conducted using both manual and automated processes. Out of three examined Machine Learning methods, two show the most promising results: Random Forest and Neural Network. Their best models performance outcomes are very close, with the Random Forest model yielding slightly superior results.

All of our experiments were performed using R software (Version 1.3.1093).

