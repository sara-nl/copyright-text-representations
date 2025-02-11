from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

x = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

y = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
x_tfidf = vectorizer.fit_transform(x)

model = XGBClassifier(random_state=42)
model.fit(x_tfidf, y)

X_new = [
    "This is a new document.",
    "Yet another document for testing.",
]

X_new_tfidf = vectorizer.transform(X_new)
predictions = model.predict(X_new_tfidf)

print("Predictions:", predictions)
