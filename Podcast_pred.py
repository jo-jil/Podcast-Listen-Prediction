import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Load the data
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')

# Save target and drop it from training features
target = 'Listening_Time_minutes'
y = train[target]
X = train.drop(columns=[target])

# Combine train and test for consistent preprocessing
X['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([X, test], axis=0)

# Handle categorical features
categorical_cols = combined.select_dtypes(include='object').columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))
    label_encoders[col] = le

# Handle missing values by filling numeric columns with their mean
numeric_cols = combined.select_dtypes(include=np.number).columns
combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].mean())

# Split back into train and test
X_encoded = combined[combined['is_train'] == 1].drop(columns=['is_train'])
test_encoded = combined[combined['is_train'] == 0].drop(columns=['is_train'])

# Train model using XGBoost
model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
model.fit(X_encoded.drop(columns=['id']), y)

# Predict on test set
preds = model.predict(test_encoded.drop(columns=['id']))

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': preds
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

print("✅ Submission preview:")
print(submission.head())
