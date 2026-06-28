import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv")

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["message"])

# Labels
y = data["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# User input
email = input("Enter an email message: ")

# Predict
email_vector = vectorizer.transform([email])
prediction = model.predict(email_vector)

print("\nPrediction:", prediction[0].upper())