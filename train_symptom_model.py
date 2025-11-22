# ===============================================
# 🧠 Train Symptom-to-Disease Classification Model
# ===============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Step 1: Load dataset
data = pd.read_csv("symptom_disease.csv")
data.columns = data.columns.str.strip()  # remove any extra spaces in column names

# Step 2: Clean data (handle missing or float values safely)
data['Symptoms'] = data['Symptoms'].fillna('').astype(str)
data['Symptoms'] = data['Symptoms'].apply(lambda x: x.replace(',', ' '))

# Step 3: Prepare training data
X = data['Symptoms']
y = data['Disease']

# Step 4: Convert text to numeric vectors
vectorizer = TfidfVectorizer()
X_vectors = vectorizer.fit_transform(X)

# Step 5: Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_vectors, y)

# Step 6: Save model and vectorizer
joblib.dump(model, "symptom_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer trained & saved successfully!")
