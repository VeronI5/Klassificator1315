import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import nltk

nltk.download('brown')

# Загрузим датасет
from nltk.corpus import brown

categories = ['learned', 'humor', 'reviews', 'news', 'fiction', 'editorial']
data = []
for category in categories:
    fileids = brown.fileids(categories=category)
    for fileid in fileids:
        article = brown.raw(fileid)
        data.append([article, category])

# Добавим дополнительные статьи для новых категорий
online_shop_articles = [
    "Buy cheap laptops and computer accessories at our online store.",
    "We have a wide range of products including electronics, clothing, and home goods.",
    "Get the best deals on our online shop today!"
]
promo_site_articles = [
    "Discover the latest fashion trends and find the perfect outfit for any occasion.",
    "Looking for a great deal? Check out our promo site for exclusive discounts and offers!",
    "Stay up to date on the latest technology and get the best deals on our promo site."
]
for article in online_shop_articles:
    data.append([article, 'online-shop'])
for article in promo_site_articles:
    data.append([article, 'promo-site'])

df = pd.DataFrame(data, columns=['text', 'theme'])

# Разделяем данные на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['theme'], test_size=0.2, random_state=42)

# Создаем pipeline, который будет состоять из векторайзера и модели
models = [
    ('MultinomialNB', Pipeline([('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 3))), ('model', MultinomialNB())])),
    ('LogisticRegression', Pipeline([('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 3))), ('model', LogisticRegression(max_iter=200))])),
    ('LinearSVC', Pipeline([('vectorizer', CountVectorizer(stop_words='english', ngram_range=(1, 3))), ('model', LinearSVC())]))
]

for name, model in models:
    # Обучаем модель
    model.fit(X_train, y_train)

    # Оцениваем точность модели на тестовом наборе
    accuracy = model.score(X_test, y_test)
    print(f'{name} accuracy: {accuracy:.3f}')

# Сохраняем модель и векторайзер в файл
import joblib

joblib.dump(models[0][1], 'model.joblib')
joblib.dump(models[0][1]['vectorizer'], 'vectorizer.joblib')

# Загружаем модель и векторайзер из файла
loaded_model = joblib.load('model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

# Используем загруженную модель для предсказаний
new_texts = ["This is a test.", "Another test."]
predictions = loaded_model.predict(new_texts)

print("Predictions:", predictions)
