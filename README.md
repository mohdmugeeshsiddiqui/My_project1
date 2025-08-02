# My_project1
# Sentiment Analysis of Product Reviews

This project performs sentiment analysis on product reviews using Natural Language Processing (NLP) and machine learning techniques in Python.

## 📌 Objective
To determine whether a given product review is **Positive** or **Negative** based on its text content.

## 🧰 Tools & Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NLTK
- Scikit-learn
- Matplotlib / Seaborn
- WordCloud

## 📂 Dataset
The dataset used is a collection of Amazon product reviews. Key columns:
- `Text`: Review text
- `Score`: Numerical rating (used to derive sentiment)

## ✅ Sentiment Labelling
- Reviews with score >= 4 are labeled **Positive**
- Reviews with score <= 2 are labeled **Negative**
- Neutral reviews (score == 3) are ignored

## 🛠️ Steps Performed
1. Load the dataset and clean the data
2. Preprocess text using tokenization, stopword removal
3. Label the data as positive or negative
4. Split into training and test datasets
5. Train a Naive Bayes classifier
6. Predict and evaluate accuracy
7. Visualize results using word clouds and plots

## 📈 Output
- Model accuracy: ~85%
- Classification of reviews
- WordCloud of positive and negative words

## 💡 Conclusion
This project helps in understanding user sentiment from product reviews and can be extended to other domains like movie reviews, tweets, etc.

---

*Created for learning and demonstration purposes.*
