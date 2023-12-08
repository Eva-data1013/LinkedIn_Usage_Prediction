import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Function to clean data
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Load dataset
df = pd.read_csv("social_media_usage.csv")

# Preprocess the data
df["sm_li"] = clean_sm(df["web1h"])
df["income"] = np.where(df["income"] > 9, np.nan, df["income"])
df["education"] = np.where(df["educ2"] > 8, np.nan, df["educ2"])
df["parent"] = clean_sm(df["par"])
df["married"] = np.where(df["marital"] == 1, 1, 0)
df["female"] = np.where(df["gender"] == 2, 1, np.where(df["gender"] == 1, 0, np.nan))
df["age"] = np.where(df["age"] > 98, np.nan, df["age"])

# Drop any rows with missing values
df.dropna(inplace=True)

# Split the data into features (X) and target (y)
X = df[['income', 'education', 'parent', 'married', 'female', 'age']]
y = df['sm_li']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate the model
y_pred = lr_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(lr_model, 'trained_lr_model.joblib')

print("Model training and evaluation complete. The model is saved.")
