import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.metrics import precision_recall_curve


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from xgboost import XGBClassifier, plot_importance

# ==================== LOAD DATA ====================

print("Loading dataset...")
df = pd.read_csv(r"D:\Credit_Card_Fraud_detection\creditcard.csv")


# =================== Imbalance check ===================
print(df['Class'].value_counts())
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (Fraud vs Non-Fraud)")
plt.show()

# Features and target
X = df.drop("Class", axis=1)   #Independent variable
y = df["Class"]               #Dependent variable





# ==================== TRAIN TEST SPLIT ====================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train[["Amount", "Time"]] = scaler.fit_transform(
    X_train[["Amount", "Time"]]
)

X_test[["Amount", "Time"]] = scaler.transform(
    X_test[["Amount", "Time"]]
)

X_train = X_train.astype("float32")
X_test  = X_test.astype("float32")


# ==================== XGBOOST MODEL (Optimized for Speed & Accuracy) ====================

fraud_ratio = y_train.mean()
scale_weight = (1 - fraud_ratio) / fraud_ratio

# Detect GPU if available (requires CUDA installed)
tree_method = "hist"

print(f" Using XGBoost with tree_method = '{tree_method}'")


xgb = XGBClassifier(
    max_depth=5,                  # shallower trees = faster training
    learning_rate=0.1,            # slightly higher learning rate
    n_estimators=200,             # fewer trees but efficient
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    random_state=42,
    tree_method="hist",      
    eval_metric="auc",
    use_label_encoder=False
)

print(" Training XGBoost model...")
xgb.fit(X_train, y_train)


print(" XGBoost training complete!")

# ==================== PREDICTIONS ====================

xgb_probs = xgb.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, xgb_probs)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
best_threshold = thresholds[np.argmax(f1_scores)]
print("Best F1-optimized threshold:", best_threshold)
xgb_preds = (xgb_probs >= best_threshold).astype(int)



 
# ==================== EVALUATION ====================

print("\n=====  XGBoost Results =====")
print(confusion_matrix(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds, digits=4))


# ==================== ROC CURVE ====================

fpr, tpr, _ = roc_curve(y_test, xgb_probs)
auc_score = roc_auc_score(y_test, xgb_probs)
print("ROC-AUC:", roc_auc_score(y_test, xgb_probs))


# ==================== CONFUSION MATRIX PLOTTING FUNCTION ====================
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==================== ROC CURVE PLOT ====================


plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"XGBoost AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Model")
plt.legend()
plt.tight_layout()
plt.show()


# ===================== PRECISION-RECALL-CURVE =====================

precision, recall, _ = precision_recall_curve(y_test, xgb_probs)

plt.figure(figsize=(6,4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.tight_layout()
plt.show()

# ==================== FEATURE IMPORTANCE ====================

plt.figure(figsize=(8, 6))
plot_importance(xgb, max_num_features=10)
plt.title("Top 10 Important Features (XGBoost)")
plt.tight_layout()
plt.show()

# ==================== SAVE MODELS ====================

joblib.dump(xgb, "xgboost_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\n Models saved successfully for deployment!")
    











