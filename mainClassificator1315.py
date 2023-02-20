import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import pandas as pd
from urllib.parse import urlparse
import validators
import sqlite3
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from selenium.webdriver.chrome.service import Service

# создаем главное окно приложения
root = tk.Tk()
root.title("Классификация сайтов")
root.minsize(width=500, height=300)
root.geometry("500x300")

# создаем базу данных для хранения классифицированных данных
conn = sqlite3.connect('classified_websites.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS websites (url TEXT, theme TEXT)''')
conn.commit()

# загружаем модель машинного обучения
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

# создаем функцию для классификации сайта
def classify_website():
    # получаем ссылку на сайт из текстового поля
    url = url_entry.get()

    # проверяем, что ссылка является валидным URL
    if not validators.url(url):
        messagebox.showerror("Ошибка", "Введенная ссылка не является корректным URL")
        return

    # загружаем страницу в браузере (нужен браузер Chrome)
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)

    # извлекаем текст со страницы
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    text = soup.get_text()

    # используем модель машинного обучения для классификации сайта
    vectorized_text = vectorizer.transform([text]).toarray()
    predicted_theme = model.predict(vectorized_text)[0]

    # сохраняем результат в базу данных
    c.execute("INSERT INTO websites (url, theme) VALUES (?, ?)", (url, predicted_theme))
    conn.commit()

    # выводим сообщение с результатом классификации
    messagebox.showinfo("Классифицирован", predicted_theme)

# создаем функцию для выгрузки данных в формате CSV
def export_data():
    # получаем все данные из базы данных
    c.execute("SELECT * FROM websites")
    rows = c.fetchall()

    # сохраняем данные в файле CSV
    df = pd.DataFrame(rows, columns=['URL', 'Тема'])
    df.to_csv('classified_websites.csv', index=False)

    # выводим сообщение об успешной экспорте
    messagebox.showinfo("Данные выгружены", "Данные были выгружены в файл classified_websites.csv")

# создаем функцию для подгрузки и классификации данных из файла
def classify_file():
    # открываем диалоговое окно для выбора файла
    file_path = filedialog.askopenfilename()

    #загружаем данные из файла
    df = pd.read_csv(file_path)

    # добавляем новый столбец 'Тема', в который будем записывать результаты классификации
    df['Тема'] = ''

    # используем модель машинного обучения для классификации каждой записи
    for i, row in df.iterrows():
        # извлекаем текст со страницы
        url = row['URL']
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        text = soup.get_text()

        # классифицируем текст
        vectorized_text = vectorizer.transform([text]).toarray()
        predicted_theme = model.predict(vectorized_text)[0]

        # записываем результаты классификации в датафрейм
        df.loc[i, 'Тема'] = predicted_theme

        # сохраняем результаты классификации в базе данных
        c.execute("INSERT INTO websites (url, theme) VALUES (?, ?)", (url, predicted_theme))
        conn.commit()

    # сохраняем классифицированные данные в файле CSV
    df.to_csv('classified_websites.csv', index=False)

    # выводим сообщение об успешной классификации и экспорте данных
    messagebox.showinfo("Файл классифицирован", f"Данные были классифицированы и выгружены в файл classified_websites.csv. Количество записей: {len(df)}")

#создаем функцию для вывода метрик качества на тестовой выборке
def test_model():
    # загружаем тестовую выборку
    test_df = pd.read_csv('test_data.csv')
    # классифицируем каждый текст из тестовой выборки
    predicted_labels = []
    for text in test_df['text']:
        vectorized_text = vectorizer.transform([text]).toarray()
        predicted_theme = model.predict(vectorized_text)[0]
        predicted_labels.append(predicted_theme)

    # вычисляем метрики качества
    actual_labels = test_df['theme']
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision = precision_score(actual_labels, predicted_labels, average='macro')
    recall = recall_score(actual_labels, predicted_labels, average='macro')
    f1 = f1_score(actual_labels, predicted_labels, average='macro')

    # выводим метрики в сообщении
    messagebox.showinfo("Метрики качества", f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}")

import subprocess

def train_model():
    subprocess.call(['python', 'Машинное обучение.py'])

# создаем текстовое поле для ввода ссылки на сайт
url_label = tk.Label(text="Введите ссылку на сайт:")
url_label.pack()
url_entry = tk.Entry(root, width=75)
url_entry.pack()

# создаем кнопку для классификации сайта
classify_button = tk.Button(text="Классифицировать", command=classify_website)
classify_button.pack()

# создаем кнопку для выгрузки данных в формате CSV
export_button = tk.Button(text="Выгрузить данные", command=export_data)
export_button.pack()

# создаем кнопку для загрузки данных из файла и их классификации
classify_file_button = tk.Button(text="Классифицировать данные из файла", command=classify_file)
classify_file_button.pack()

# создаем кнопку для вывода метрик качества на тестовой выборке
test_model_button = tk.Button(text="Тест модели", command=test_model)
test_model_button.pack()

train_button = tk.Button(root, text="Обучить", command=train_model)
train_button.pack()

root.mainloop()

