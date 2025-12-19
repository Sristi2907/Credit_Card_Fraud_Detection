# Credit_Card_Fraud_Detection
Credit Card Fraud Detection using Machine Learning (XGBoost) with class imbalance handling, ROC-AUC and Precision-Recall based threshold optimization.
# Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.  
The dataset is highly imbalanced, so advanced evaluation metrics and threshold optimization are used.

## 🔹 Dataset
- Credit Card Transactions Dataset
- Features are PCA-transformed (except Time and Amount)
- Highly imbalanced (Fraud ≪ Normal)

## 🔹 Methodology
- Data preprocessing and scaling
- Stratified train-test split
- Model: XGBoost Classifier
- Class imbalance handled using `scale_pos_weight`
- Evaluation using:
  - ROC-AUC
  - Precision-Recall Curve
- Optimal classification threshold selected by maximizing F1-score

## 🔹 Results
- High ROC-AUC score
- Improved fraud detection using optimized threshold
- Balanced trade-off between precision and recall

## 🔹 Technologies Used
- Python
- NumPy, Pandas
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## 🔹 How to Run
```bash
pip install -r requirements.txt
python fraud_detection.py

