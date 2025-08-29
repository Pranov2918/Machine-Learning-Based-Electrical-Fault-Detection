import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the datasets
class_data_path = r"C:\Users\Pranov\OneDrive\Desktop\Fault Detection\classData.csv"
detect_data_path = r"C:\Users\Pranov\OneDrive\Desktop\Fault Detection\detect_dataset.csv"

class_df = pd.read_csv(class_data_path)
detect_df = pd.read_csv(detect_data_path)

# Preview the data
print("Class Data Sample:")
print(class_df.head())
print("\nDetect Data Sample:")
print(detect_df.head())

# Check for missing values
print("\nMissing values in classData.csv:")
print(class_df.isnull().sum())
print("\nMissing values in detect_dataset.csv:")
print(detect_df.isnull().sum())

# Select features and labels from the detect dataset for classification
X = detect_df[['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']]
y = detect_df['Output (S)']  # Fault labels (0 or 1 or other classes)

# Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("\nShape of training set:", X_train.shape)
print("Shape of testing set:", X_test.shape)
