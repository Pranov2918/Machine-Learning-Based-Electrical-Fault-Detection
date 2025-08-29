import pandas as pd
import numpy as np
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
detect_data_path = r"C:\Users\Pranov\OneDrive\Desktop\Fault Detection\detect_dataset.csv"
df = pd.read_csv(detect_data_path)

# Initialize features DataFrame
features = pd.DataFrame()

# Rolling window size
window_size = 10

# Compute time-domain rolling features properly using pandas rolling methods
for col in ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']:
    features[col + '_mean'] = df[col].rolling(window=window_size).mean()
    features[col + '_std'] = df[col].rolling(window=window_size).std()
    features[col + '_skew'] = df[col].rolling(window=window_size).skew()
    features[col + '_kurt'] = df[col].rolling(window=window_size).kurt()  # Correct kurtosis calculation

# Drop NA values resulting from rolling window calculations
features = features.dropna().reset_index(drop=True)
target = df['Output (S)'][-len(features):].reset_index(drop=True)

# Frequency-domain features using FFT rolling window
fft_window = 64

for col in ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']:
    features[col + '_fft_mean'] = df[col].rolling(window=fft_window)\
        .apply(lambda x: np.abs(fft(x))[:fft_window//2].mean(), raw=True)

# Drop NA values resulting from FFT rolling
features = features.dropna().reset_index(drop=True)
target = target[-len(features):].reset_index(drop=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, test_size=0.2, random_state=42)

# Train Random Forest with tuned hyperparameters
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42
)
rf.fit(X_train, y_train)

# Predict and evaluate on test data
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Feature Engineering')
plt.show()
