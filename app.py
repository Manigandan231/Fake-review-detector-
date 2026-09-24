import joblib
import re

# ==========================================
# FAKE REVIEW DETECTION APPLICATION
# ==========================================

# Load trained model and TF-IDF vectorizer
model = joblib.load("model/fake_review_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# ==========================================
# DISPLAY APPLICATION TITLE
# ==========================================

print("=" * 50)
print("          FAKE REVIEW DETECTION")
print("       Machine Learning + NLP")
print("=" * 50)


# ==========================================
# GET REVIEW FROM USER
# ==========================================

review = input("\nEnter your review: ")


# ==========================================
# PREPROCESS THE REVIEW
# ==========================================

cleaned_review = clean_text(review)


# ==========================================
# TF-IDF TRANSFORMATION
# ==========================================

review_vector = vectorizer.transform([cleaned_review])


# ==========================================
# PREDICT REVIEW
# ==========================================

prediction = model.predict(review_vector)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "-" * 50)
print("Review:", review)
print("-" * 50)

if prediction[0] == 1:
    print("Prediction: FAKE REVIEW")
else:
    print("Prediction: GENUINE REVIEW")

print("-" * 50)
print("Prediction completed successfully!")
print("=" * 50)
