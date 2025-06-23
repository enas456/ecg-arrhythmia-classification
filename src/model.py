import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter

# === Load the balanced training set ===
train_path = r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\ECG Signal Processing and Segmentation\train_balanced_8000.xlsx'
train_df = pd.read_excel(train_path)
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].astype(str).values

# === Load the imbalanced test set ===
test_path = r'C:\Users\ENAS.DESKTOP-CSI0DAD\Desktop\ecg\me\dataset\ECG Signal Processing and Segmentation\test_imbalanced.xlsx'
test_df = pd.read_excel(test_path)
X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].astype(str).values

# === Show class distributions ===
print(f"Train class distribution: {Counter(y_train)}")
print(f"Test class distribution: {Counter(y_test)}")

# === Train the Random Forest classifier with class_weight='balanced' ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# === Predict on test data ===
y_pred = clf.predict(X_test)

# === Evaluate the model ===
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=['F', 'N', 'Q', 'S', 'V'])  # consistent label order
class_report = classification_report(y_test, y_pred, labels=['F', 'N', 'Q', 'S', 'V'])

# === Print results ===
print(f"\nAccuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# === Confusion matrix visualization ===
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['F', 'N', 'Q', 'S', 'V'],
            yticklabels=['F', 'N', 'Q', 'S', 'V'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
