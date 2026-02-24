# text-analytics-for-spam-detection-using-naive-bayes-
" import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA (Example structure: 'text' column and 'label' column)
# data = pd.read_csv('spam.csv') 
# For demo, let's use a mini-dataset:
data = pd.DataFrame({
    'text': [
        'Get a free cruise now!', 'Hey, are we still meeting?',
        'WIN CASH PRIZE TODAY', 'Can you send me the report?',
        'Congratulations, you won a gift card', 'Dinner at 7 tonight?'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
})

# 2. TEXT VECTORIZATION
# This converts text into a matrix of token counts
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(data['text'])
y = data['label']

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. INITIALIZE & TRAIN NAIVE BAYES
# MultinomialNB is best for discrete counts (like word frequencies)
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. PREDICTION & EVALUATION
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))

# 6. TEST ON A NEW MESSAGE
new_msg = ["You have won a $1000 Walmart coupon! Click here."]
new_msg_count = vectorizer.transform(new_msg)
result = model.predict(new_msg_count)
print(f"The message is: {result[0]}") "

"OUTPUT"
Accuracy: 0.5
              precision    recall  f1-score   support

         ham       0.50      1.00      0.67         1
        spam       0.00      0.00      0.00         1

    accuracy                           0.50         2
   macro avg       0.25      0.50      0.33         2
weighted avg       0.25      0.50      0.33         2

The message is: spam
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.12/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
