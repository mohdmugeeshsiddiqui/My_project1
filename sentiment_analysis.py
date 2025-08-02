# 📦 Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 📂 Step 2: Load CSV file
data = pd.read_csv("product_reviews.csv")  # ✅ Make sure this file is in the same folder

# 👀 Step 3: Convert 'positive'/'negative' to 1/0
data['Sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# 🎯 Step 4: Define input (X) and output (y)
X = data['review']        # Review text
y = data['Sentiment']     # 0 = Negative, 1 = Positive

# 🧠 Step 5: Convert text into numerical vectors using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# 🔀 Step 6: Split data into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 Step 7: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# 🤖 Step 8: Predict on test data
y_pred = model.predict(X_test)

# 📈 Step 9: Show accuracy and classification report
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
