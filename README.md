# Credit Risk Assessment

## Problem Statement

Financial institutions face significant exposure to credit default risk. Accurately identifying customers who are likely to default is essential to minimizing financial losses while maintaining responsible lending practices.

The objective of this project is to develop and evaluate predictive models capable of estimating the probability that a credit card client will default on their next payment. The primary challenge involves handling class imbalance and selecting evaluation metrics aligned with financial risk management.

---

## Dataset

- Source: UCI Machine Learning Repository  
- Observations: 30,000 credit card clients  
- Features: 24 financial and demographic variables  
- Target Variable: `default` (1 = default, 0 = non-default)  
- Default Rate: Approximately 22%

The dataset includes repayment history, bill statements, payment amounts, credit limits, and demographic attributes. No missing values were present.

---

## Methodology

### Exploratory Analysis

- Validated dataset structure and data types  
- Confirmed absence of missing values  
- Analyzed class distribution  
- Examined feature correlations  
- Identified repayment status variables as key predictors  

### Model Development

The following classification models were implemented and compared:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosting  

Data was split using stratified train-test sampling to preserve class proportions. Feature scaling was applied where required.

### Evaluation Strategy

Given the imbalanced nature of the dataset, model performance was evaluated using:

- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

Accuracy was not used as the primary selection metric.

### Threshold Optimization

The classification threshold was adjusted from the default 0.5 to 0.35 to improve recall for default detection. This reflects financial risk priorities where failing to identify a defaulter is more costly than incorrectly flagging a low-risk customer.

---

## Results

| Model               | ROC-AUC |
|--------------------|----------|
| Gradient Boosting  | 0.779    |
| Random Forest      | 0.756    |
| Logistic Regression| 0.708    |
| Decision Tree      | 0.614    |

After threshold adjustment on the Gradient Boosting model:

- Recall improved from approximately 0.36 to 0.46  
- F1-score improved to 0.52  
- Performance aligned more closely with credit risk objectives  

---

## Conclusion

Gradient Boosting delivered the strongest overall performance, achieving the highest ROC-AUC and best balance between precision and recall.

Threshold tuning significantly enhanced the model’s ability to detect defaulters without materially degrading stability.

Repayment history variables were identified as the most influential predictors, confirming that past payment behavior is the strongest indicator of future credit risk.

This project demonstrates a structured and business-aligned approach to credit risk modeling, emphasizing proper evaluation metrics, model comparison, and threshold optimization.
