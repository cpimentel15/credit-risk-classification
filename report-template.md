**credit-risk-classification**

The credit_risk_classification.ipynb is located in the folder of Credit_Risk

**Overview of the Analysis**
The analysis uses two machine learning algorithms to train and evaluate a model based on loan risk. We use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

The dataset covers various client financial information including loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, total debt, and loan status. The purpose of the machine learning model is to predict the clients’ loan status. This is categorized into ‘0’ (healthy loan) and ‘1’ (high-risk loan).

The model separates the y variable (the labels loan_status), and the X variable (the features) after dropping the y variable column. Then it uses the ‘value_counts’ function to check the balance of the labels variable (y).

The dataset is split into training and testing datasets by using train_test_split. Then a logistic regression model(LogisticRegression) is applied to the training data. And make a prediction using the tasting data.

For model evaluation, ‘balanced_accuracy_score’, ‘confusion_matrix’, and ‘classification_report’ are used to evaluate the model’s performance.

To improve the prediction precision, the dataset is over-sampled by using RandomOverSampler then also use a logistic regression model and three evaluation methods mentioned above to evaluate again.

**Results**

The balanced accuracy scores and the precision and recall scores of all machine learning models are as follows:

Machine Learning Model 1:
Description of Model 1 Accuracy, Precision, and Recall scores. • Accuracy: 0.9442676901753825 • Precision: 0 (healthy loan) 1.00; 1 (high-risk loan) 0.87 • Recall: 0 (healthy loan) 1.00; 1 (high-risk loan) 0.89
Machine Learning Model 2:
Description of Model 2 Accuracy, Precision, and Recall scores. • Accuracy: 0.9959744975744975 • Precision: 0 (healthy loan) 0.99; 1 (high-risk loan) 1.00 • Recall: 0 (healthy loan) 0.87; 1 (high-risk loan) 1.00

**Summary**

The logistic regression model achieves a balanced accuracy of 0.9442676901753825. The model gets 18679 correct predictions while only 80 false predictions about “healthy loan”. This indicates that the logistic regression model is capable of predicting “healthy loan” at a high precision level. Due to lower original data in “high-risk loan”. The model unfortunately predicts 558 false results in high-risk loan and the precision is 0.87. Therefore, the model gets strong precision and recall on the test dataset, especially for healthy loan prediction. This means that the model is likely to perform perfectly in real life. It can predict “healthy loan” more correctly than “high-risk loan”.

After oversampled the minority class, the model improves its precision. The balanced_accuracy_score increased to 0.9959744975744975. The recall slightly increases to 1.00 in high-risk loan. The oversampled model is able to predict healthy loan and high-risk loan very precisely.

The logistic regression model seems precise enough to predict a ‘0’ healthy loan. But if the model would like to achieve a high prediction rate in “high-risk loan”, it’d better oversample the minority class and run a logistic regression model afterward.
