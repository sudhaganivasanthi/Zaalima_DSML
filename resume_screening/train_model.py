# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import preprocess



# ✅ Load CSV from the correct path
df = pd.read_csv("data/Resumes.csv")

# ✅ Standardize column names (e.g., "Resume Text" → "resume_text")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Confirm the column is now named 'resume_text'
if "resume_text" not in df.columns or "job_role" not in df.columns:
    raise ValueError("CSV must contain 'Resume Text' and 'Job Role' columns.")

# ✅ Preprocess the text
df['cleaned'] = df['resume_text'].apply(preprocess)

# ✅ Extract features and labels
X = df['cleaned']
y = df['job_role']

# ✅ TF-IDF conversion
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# ✅ Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)



# Try different models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "RandomForest": RandomForestClassifier()
}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Save model and vectorizer
joblib.dump(model, "model/classification_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model and vectorizer saved in /model/")
