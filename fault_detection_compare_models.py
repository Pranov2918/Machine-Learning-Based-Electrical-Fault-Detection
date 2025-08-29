import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
detect_data_path = r"C:\Users\Pranov\OneDrive\Desktop\Fault Detection\detect_dataset.csv"
detect_df = pd.read_csv(detect_data_path)

# Features and target
X = detect_df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]
y = detect_df['Output (S)']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest with best hyperparameters (update params from your tuning)
rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# SVM model
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# Evaluate models
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

print("\nSVM Accuracy:", accuracy_score(y_test, svm_pred))
print("SVM Classification Report:\n", classification_report(y_test, svm_pred))

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14,6))

cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Random Forest Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_svm = confusion_matrix(y_test, svm_pred)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('SVM Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.show()
