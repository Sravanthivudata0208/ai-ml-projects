import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = {
    'message': [
        "Win money now",
        "Congratulations you won lottery",
        "Call me tonight",
        "Let's go for dinner",
        "Claim your free prize",
        "Meeting tomorrow"
    ],
    'label': [
        "spam",
        "spam",
        "ham",
        "ham",
        "spam",
        "ham"
    ]
}

df = pd.DataFrame(data)

X = df['message']
y = df['label']

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vectorized, y)

test_message = ["Free lottery prize"]

test_vector = vectorizer.transform(test_message)

prediction = model.predict(test_vector)

print("Message:", test_message[0])
print("Prediction:", prediction[0])