# 🕵️‍♂️💸 Fraud Detection Model

This repository contains a machine learning model built to detect fraudulent transactions using a real-world financial dataset. The goal is to identify fraud with high accuracy while minimizing false negatives, a critical aspect in preventing monetary losses.

---

## 📊 Project Overview

This project followed a structured ML pipeline:

1. **Data Loading & Exploratory Data Analysis (EDA):**  
   - Loaded the dataset and explored structure, distributions, and value counts.  
   - Verified data completeness (no missing values). 🔍

2. **Data Preprocessing:**  
   - Manually encoded the `type` column.  
   - Applied frequency encoding to `nameDest`.  
   - Dropped `nameOrig` and `nameDest` post-encoding.  
   - Scaled numeric features using `StandardScaler`. 🧹

3. **Model Development:**  
   - Trained an initial **Logistic Regression** model. 📈  
   - Trained an **XGBoost Classifier** for improved results. 🚀  
   - Performed **feature importance analysis** with both models. 🤔  
   - Selected key features:  
     `newbalanceOrig`, `oldbalanceOrg`, `amount`, `type`, `oldbalanceDest`, `newbalanceDest`. 🔑  
   - Retrained XGBoost on selected features. 💪  
   - Tuned hyperparameters using `RandomizedSearchCV` to optimize ROC AUC. ⚙️  
   - Trained a final, fine-tuned XGBoost model. 🎯

4. **Model Evaluation:**  
   - Evaluated models using **confusion matrix**, **classification report**, and **ROC AUC score**. ✅

5. **Threshold Adjustment:**  
   - Adjusted classification threshold to 0.45 to reduce false negatives. ⚖️

6. **Fraud Prevention Strategies:**  
   - Suggested prevention ideas based on influential features.  
   - Discussed how effectiveness can be tracked in real-world systems. 🛡️

---

## 💾 Dataset

The dataset `Fraud.csv` contains anonymized financial transactions with the following fields:

| Column            | Description                                                             |
|------------------|-------------------------------------------------------------------------|
| `step`           | Time step (1 step = 1 hour). ⏰                                          |
| `type`           | Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER). 🔄       |
| `amount`         | Transaction amount. 💰                                                   |
| `nameOrig`       | Sender's customer ID. 👤                                                |
| `oldbalanceOrg`  | Sender's balance before transaction. 🏦                                 |
| `newbalanceOrig` | Sender's balance after transaction. 💵                                 |
| `nameDest`       | Recipient's customer ID. 👥                                             |
| `oldbalanceDest` | Recipient's balance before transaction. 🏦                             |
| `newbalanceDest` | Recipient's balance after transaction. 💵                             |
| `isFraud`        | Target label (1 = fraud, 0 = genuine). 🚩                              |
| `isFlaggedFraud` | Flag by old system if suspected fraud. 🚨                              |

---

## ✨ Key Findings

- ✅ No missing values in the dataset.
- 🔄 Manual and frequency encoding were effective for categorical variables.
- 🔑 Key predictive features:  
  `newbalanceOrig`, `oldbalanceOrg`, `amount`, `type`, `oldbalanceDest`, `newbalanceDest`.
- 🏆 The **fine-tuned XGBoost model** achieved a near-perfect **ROC AUC score of 0.9996**.
- ⚖️ Adjusting classification thresholds allows control over fraud recall vs. false positive rate.

## Insights drawn
To decide on the threshold, consider:

- What is the financial cost of a false negative (missed fraud)?
- What is the operational cost of a false positive (investigating a false alarm)?
- What is the impact on customer experience of a false positive?

---

**1. Data Cleaning:**

Missing Values: Based on the output of df.isnull().sum(), there are no missing values in the dataset, so no imputation was needed for missing values. Outliers: The notebook does not explicitly address outliers. While df.describe() provides summary statistics that can indicate the presence of outliers (e.g., large differences between the 75th percentile and the max value in columns like amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, and newbalanceDest), no specific outlier detection or handling techniques were applied in the provided code. Multicollinearity: The notebook does not explicitly check for or handle multicollinearity.

**2. Fraud Detection Model Description:**

The fraud detection model developed in this notebook is an XGBoost classifier.

Initial Model: An initial XGBoost model was trained on all the features after handling categorical variables and scaling numerical features. Feature Selection: Based on the feature importances from the initial XGBoost model and the absolute coefficients from a Logistic Regression model, a subset of features was selected: 'newbalanceOrig', 'oldbalanceOrg', 'amount', 'type', 'oldbalanceDest', and 'newbalanceDest'. Retrained Model: An XGBoost model was retrained using only the selected features. Hyperparameter Tuning: Randomized Search with cross-validation was performed on the retrained XGBoost model to find the best hyperparameters to optimize the ROC AUC score. The best hyperparameters found were {'subsample': 1.0, 'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 1.0}. Final Model: A final XGBoost model was initialized and trained using the best hyperparameters found during tuning and the selected features. Threshold Adjustment: The classification threshold of the final model was adjusted to 0.45 to analyze its impact on false negatives.

**3. Variable Selection:**

Variables were selected based on the feature importances from the initial XGBoost model and the absolute coefficients from a Logistic Regression model. Features that showed consistently high importance/coefficients in both models were selected. The selected features were: 'newbalanceOrig', 'oldbalanceOrg', 'amount', 'type', 'oldbalanceDest', and 'newbalanceDest'.

**4. Model Performance Demonstration:**

The performance of the models was demonstrated using the following tools and metrics:

Confusion Matrix: Visualized using heatmaps to show the counts of true positives, true negatives, false positives, and false negatives. This is a crucial tool for understanding the types of errors the model makes. Classification Report: Provided precision, recall, f1-score, and support for both classes (fraud and not fraud). This gives a detailed breakdown of the model's performance for each class. ROC AUC Score: Calculated to measure the model's ability to distinguish between the positive and negative classes. A higher ROC AUC score indicates better performance. The performance metrics for the initial XGBoost model, the retrained XGBoost model with selected features, and the fine-tuned XGBoost model with selected features and a threshold of 0.45 were presented. The fine-tuned model showed a high ROC AUC score, indicating good overall performance.

**5. Key Factors that Predict Fraudulent Customers:**

Based on the feature importances from the XGBoost model and the absolute coefficients from the Logistic Regression model, the key factors that predict fraudulent customers are:

newbalanceOrig: New balance of the originating account after the transaction.
oldbalanceOrg: Old balance of the originating account before the transaction.
amount: The amount of the transaction.
type: The type of transaction (e.g., CASH-IN, CASH-OUT, TRANSFER). oldbalanceDest: Old balance of the destination account before the transaction. newbalanceDest: New balance of the destination account after the transaction.

**6. Do these factors make sense?**

Yes, these factors generally make sense in the context of fraud detection:

Balance Changes (newbalanceOrig, oldbalanceOrg, oldbalanceDest, newbalanceDest): Fraudulent transactions often involve significant changes in account balances, especially in the originating and destination accounts. Unusual balance behaviors can be strong indicators of fraudulent activity.
Amount (amount): Fraudulent transactions can involve unusually large or small amounts, depending on the nature of the fraud. The transaction amount is a fundamental feature in analyzing suspicious activity.
Transaction Type (type): Certain transaction types, like 'TRANSFER' and 'CASH-OUT', are more commonly associated with fraudulent activities than others like 'PAYMENT' or 'CASH-IN'.
Features like step, nameDest_freq, and isFlaggedFraud were found to be less important, which also makes sense. step is a time step and might not be directly indicative of fraud itself, but rather the timing within the simulation. nameDest_freq being less important suggests that while frequency of transactions to a destination might play a minor role, it's not as strong a predictor as the balance and amount features. isFlaggedFraud is a flag set by the system and might not capture all types of fraud or could be based on simpler rules that the model can learn from other features.

**7. Prevention Strategies:**

Based on the analysis and model, prevention strategies should focus on the key factors identified:

Real-time Monitoring of Balance Changes: Implement systems that monitor significant and unusual changes in both originating and destination account balances, particularly for 'TRANSFER' and 'CASH-OUT' transactions.
Transaction Amount Thresholds and Anomaly Detection: Set up alerts or blocks for transactions exceeding certain thresholds or those that deviate significantly from a customer's typical transaction patterns.
Behavioral Analysis: Develop models that analyze customer behavior and flag transactions that are inconsistent with their historical activity (e.g., large transfers to new destinations, unusual transaction types).
Two-Factor Authentication for High-Risk Transactions: Require additional verification for transactions that are flagged as potentially fraudulent based on the model's predictions.
Transaction Type-Specific Rules: Implement stricter rules and monitoring for 'TRANSFER' and 'CASH-OUT' transactions.
When updating infrastructure, the company should prioritize building a robust real-time transaction monitoring system that can integrate the developed fraud detection model. This infrastructure should be scalable to handle the volume of transactions and allow for quick responses to potential fraud.

**8. Determining if Prevention Strategies Work:**

To determine if the implemented prevention strategies are working, the company should:

Track Key Performance Indicators (KPIs): Monitor metrics such as the number of fraudulent transactions detected and prevented, the value of prevented fraud, the false positive rate, and the false negative rate.

A/B Testing: If possible, implement new strategies on a subset of transactions or customers to compare their effectiveness against the existing system.

Customer Feedback: Collect feedback from customers regarding false positives (transactions incorrectly flagged as fraudulent) to understand the impact on user experience.

Manual Review and Investigation: Continue to have a process for manually reviewing flagged transactions to identify new fraud patterns that the model might not yet capture.

Model Retraining and Monitoring: Regularly retrain the fraud detection model with new data that includes the outcomes of implemented prevention strategies. Monitor the model's performance over time to ensure it remains effective as fraud tactics evolve.

Analyze False Negatives: Continue to analyze false negatives to understand why these fraudulent transactions were missed and use this information to refine both the model and the prevention strategies.

---

## 📈 Model Performance

### 🔹 Fine-Tuned XGBoost Model (Threshold = 0.45)

- **Confusion Matrix:**  
<img width="680" height="393" alt="image" src="https://github.com/user-attachments/assets/f37bbcff-5126-4d56-938d-3e42ca255f89" />


- **Classification Report:**  
Confusion Matrix (Fine-tuned XGBoost with selected features and threshold=0.45):
[[1270829      52]
 [    390    1253]]

Classification Report (Fine-tuned XGBoost with selected features and threshold=0.45):
              precision    recall  f1-score   support

          0       1.00      1.00      1.00   1270881
          1       0.96      0.76      0.85      1643

    accuracy                           1.00   1272524
    
   macro avg       0.98      0.88      0.92   1272524
   
weighted avg       1.00      1.00      1.00   1272524

ROC AUC Score (Fine-tuned XGBoost with selected features): 0.9996428991988627


---

### 🔹 Logistic Regression (Baseline Model)

- **Confusion Matrix:**
[[1228088   42793]
 [    104    1539]]
              precision    recall  f1-score   support

           0       1.00      0.97      0.98   1270881
           1       0.03      0.94      0.07      1643

    accuracy                           0.97   1272524
  
   macro avg       0.52      0.95      0.52   1272524
  
weighted avg       1.00      0.97      0.98   1272524

ROC AUC Score: 0.9515146195331061

---

✅ Despite low precision for frauds, Logistic Regression had excellent **recall (0.94)** — crucial for flagging potential fraud, even at the cost of more false positives.

---

## 🚀 Future Enhancements

- 🛠️ Deploy model via Flask/FastAPI for real-time predictions.
- 📡 Integrate with Kafka or AWS Kinesis for streaming detection.
- 🤖 Experiment with LSTM or Autoencoder-based anomaly detection.
- 📊 Use SHAP or LIME for deeper model explainability.

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io)  
- [Scikit-learn Documentation](https://scikit-learn.org)
- [Dataset link](https://www.kaggle.com/datasets/amanindiamuz/financial-dataset-for-fraud-detection-in-a-comapny)


---

## 🙌 Acknowledgments

Thanks to the mentors, datasets providers, and open-source contributors who made this project possible.

---
