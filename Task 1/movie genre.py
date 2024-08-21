# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 18:51:58 2024

@author: uzman
"""

import re
import string
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


train_path = "train_data.txt"
train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

train_data.columns

train_data.head(10)

test_path = "test_data.txt"
test_data = pd.read_csv(test_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')

test_data.head(20)




# Encode the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['Genre'])

test_solution_path = "test_data_solution.txt"
test_solution = pd.read_csv(test_solution_path, sep=":::", names=['Id', 'Title', 'Genre', 'Description'], engine="python")

test_solution.head(10)

genre = train_data['Genre'].value_counts()
genre = genre.sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre.index, y=genre.values)
plt.title('Distribution of Genres in Training Data', fontsize=16)
plt.xlabel('Genre', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()




import nltk
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import re
import string

stemmer = LancasterStemmer()
stop_words = set(stopwords.words("english"))  # Stopwords set

def cleaning_data(text):
    text = text.lower()
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'.pic\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Change to replace non-characters with a space
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    text = " ".join([i for i in words if i not in stop_words and len(i) > 2])
    text = re.sub(r"\s+", " ", text).strip()  # Replace multiple spaces with a single space
    return text

train_data['clean_description'] = train_data['Description'].apply(cleaning_data)
test_data['clean_description'] = test_data['Description'].apply(cleaning_data)
test_solution['clean_description'] = test_solution['Description'].apply(cleaning_data)


# Define the pipeline for SVC
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_df=0.9, ngram_range=(1, 2))),
    ('model', SVC(kernel='linear', C=1))
])


X_train, X_test, y_train, y_test = train_test_split(train_data['clean_description'], y_train, test_size=0.2, random_state=42)


# Train and evaluate the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_train_pred = pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Validation Accuracy: {test_accuracy}")


# Test with the actual test data
test_descriptions = test_data['clean_description']
test_predictions = pipeline.predict(test_descriptions)
test_predictions_labels = label_encoder.inverse_transform(test_predictions)

# Add predictions to test data
test_data['Predicted_Gener'] = test_predictions_labels
# Compare predictions with the actual genres from the solution
test_solution = test_solution[['Id', 'Gener']]
test_data = test_data[['Id', 'Predicted_Gener']]
comparison_df = test_data.merge(test_solution, on='Id')
comparison_df.columns = ['Id', 'Predicted_Gener', 'Actual_Gener']

# Calculate accuracy
accuracy = accuracy_score(comparison_df['Actual_Gener'], comparison_df['Predicted_Gener'])
print(f"Test Data Accuracy: {accuracy}")