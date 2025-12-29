import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords (first time only)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv(r"C:\Users\Manikandansaravanan\OneDrive\Documents\Fake-review-detector--main\data\fake_reviews_dataset.csv")

# Select columns
texts = data['text']
labels = data['label']

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Apply cleaning
texts = texts.apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    max_features=5000
)

X = vectorizer.fit_transform(texts)

print("Data preprocessing completed")
print("Feature shape:", X.shape)


import pandas as pd
import nltk
import re
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv(r"C:\Users\Manikandansaravanan\OneDrive\Documents\Fake-review-detector--main\data\fake_reviews_dataset.csv")

# Select columns
texts = data['text']
labels = data['label']

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Clean text
texts = texts.apply(clean_text)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words('english'),
    max_features=5000
)

X = vectorizer.fit_transform(texts)
y = labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model (Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model training completed")
print("Accuracy:", accuracy)

# Save model and vectorizer
joblib.dump(model, r"C:\Users\Manikandansaravanan\OneDrive\Documents\Fake-review-detector--main\model\fake_review_model.pkl")
joblib.dump(vectorizer, r"C:\Users\Manikandansaravanan\OneDrive\Documents\Fake-review-detector--main\model\vectorizer.pkl")

print("Model and vectorizer saved successfully")
