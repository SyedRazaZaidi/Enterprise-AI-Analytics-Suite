import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("⏳ Loading real-world Superstore dataset...")

# 1. Load the dataset
data_path = os.path.join(os.path.dirname(__file__), '../data/superstore.csv')
df = pd.read_csv(data_path, encoding='windows-1252') # Common encoding for this dataset

# 2. Data Cleaning & Feature Engineering
print("🧹 Cleaning data and engineering features...")
# Select only the features we need
df = df[['Region', 'Quantity', 'Discount', 'Sales']]

# Drop any missing values to prevent model crashing
df = df.dropna()

# Convert the text 'Region' into numbers the ML model can understand
region_mapping = {'East': 0, 'West': 1, 'Central': 2, 'South': 3}
df['Region_Code'] = df['Region'].map(region_mapping)

# Define our Features (X) and Target (y)
X = df[['Quantity', 'Discount', 'Region_Code']]
y = df['Sales']

# Split data into training and testing sets to prove accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the Model
print("🧠 Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the Model (Great for your defense!)
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"📊 Model Accuracy (R2 Score): {r2:.2f}")
print(f"📉 Mean Absolute Error: ${mae:.2f}")

# 5. Save the trained model to disk
model_path = os.path.join(os.path.dirname(__file__), 'sales_model.pkl')
joblib.dump(model, model_path)

print(f"✅ Real-world model successfully trained and saved to: {model_path}")