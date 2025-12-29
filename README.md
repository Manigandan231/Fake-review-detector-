## Fake Review Detection using Machine Learning
📌 Project Overview

Online reviews play a crucial role in influencing customer decisions. However, the presence of fake and misleading reviews reduces trust in online platforms. This project focuses on detecting fake reviews using Natural Language Processing (NLP) and Machine Learning techniques such as TF-IDF and classification algorithms.

## 📅 Day 1: Problem Study & Planning
🔍 Objective

To understand the problem of fake review detection and plan the project workflow.

🛠 Activities Performed

Studied the concept of fake and genuine reviews

Analyzed the impact of fake reviews on e-commerce platforms

Defined project scope and objectives

Selected tools and technologies:

Python

Natural Language Processing (NLP)

Machine Learning models

Prepared a 7-day project plan and abstract

✅ Outcome

✔ Clear understanding of the problem
✔ Project plan and abstract finalized

## 📅 Day 2: Dataset Collection
🔍 Objective

To collect and understand a labeled dataset for fake review detection.

🛠 Activities Performed

Downloaded a fake review dataset from Kaggle

Studied dataset structure and labels

Prepared dataset with two columns:

review_text – review content

label – 0 (Genuine), 1 (Fake)

Stored the dataset as reviews.csv in the data/ folder

📂 Dataset Location
data/reviews.csv

✅ Outcome

✔ Dataset successfully collected and organized
✔ Ready for preprocessing

## 📅 Day 3: Data Preprocessing
🔍 Objective

To clean and preprocess textual review data and convert it into numerical features.

🛠 Activities Performed

Loaded dataset using Pandas

Cleaned review text by:

Converting to lowercase

Removing punctuation and special characters

Removed English stopwords using NLTK

Applied TF-IDF Vectorization to transform text into numerical format

📚 Techniques Used

Natural Language Processing (NLP)

Text Cleaning

Stopword Removal

TF-IDF (Term Frequency–Inverse Document Frequency)

📂 Files Updated

train.py

✅ Outcome

✔ Cleaned and structured data prepared
✔ Feature matrix ready for model training
