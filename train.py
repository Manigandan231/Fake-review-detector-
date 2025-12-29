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

