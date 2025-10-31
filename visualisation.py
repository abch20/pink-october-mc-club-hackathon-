# ============================================================
# 🧠 Brain Tumor Stage Classification - Random Forest Upgrade
# ============================================================

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def find_local_csv(filename: str) -> Path:
    """Find CSV file in common local locations."""
    candidates = [
        Path(filename),
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        Path(__file__).resolve().parent / 'data' / filename,
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

train_path = find_local_csv('train.csv')
test_path = find_local_csv('test.csv')

if train_path is None or test_path is None:
    print("❌ Could not find 'train.csv' or 'test.csv'!", file=sys.stderr)
    sys.exit(2)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("✅ Data loaded successfully!")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)



for col in ['id', 'ID', 'Id']:
    train_df.drop(columns=[col], errors='ignore', inplace=True)
    test_df.drop(columns=[col], errors='ignore', inplace=True)

train_df = train_df.fillna(train_df.median(numeric_only=True))
test_df = test_df.fillna(test_df.median(numeric_only=True))

target_col = 'cancer_stage'
feature_cols = [
    'size', 'location', 'edema', 'necrosis', 'enhancement', 'shape',
    'margins', 'calcification', 'cystic_components', 'hemorrhage',
    'ki67_index', 'mitotic_count', 'age'
]
feature_cols = [c for c in feature_cols if c in train_df.columns]

X = train_df[feature_cols].copy()
y = train_df[target_col].copy()


for col in X.columns:
    if X[col].dtype == 'object' or isinstance(X[col].iloc[0], str):
        print(f"⚙️ Encoding categorical column: {col}")
        le = LabelEncoder()
        combined = pd.concat([X[col], test_df[col]], axis=0).astype(str)
        le.fit(combined)
        X[col] = le.transform(X[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))



X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Validation size: {X_val.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df[feature_cols])



model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)


cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_macro')
print(f"\n🧩 Cross-validation Macro F1-scores: {cv_f1}")
print(f"➡️ Mean CV F1-score: {cv_f1.mean():.4f}")


y_pred = model.predict(X_val_scaled)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='macro')
recall = recall_score(y_val, y_pred, average='macro')
f1_manual = 2 * (precision * recall) / (precision + recall)
f1_macro = f1_score(y_val, y_pred, average='macro')

print("\n📈 Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro, sklearn): {f1_macro:.4f}")
print(f"F1-score (manual formula): {f1_manual:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred))



cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()



importances = model.feature_importances_
feat_imp = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(data=feat_imp.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

print("\n🔍 Top 10 Important Features:")
print(feat_imp.head(10))



test_pred = model.predict(X_test_scaled)
submission = pd.DataFrame({
    "id": np.arange(len(test_pred)),
    "Stage": test_pred
})  
submission.to_csv("submission_rf.csv", index=False)

print("\n✅ Submission file created successfully: submission_rf.csv")
