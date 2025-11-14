import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
# Загрузка данных
data = pd.read_csv('candy_data.csv')

# Разделение данных на признаки и целевую переменную
X = data[['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent']]
y = data['competitorname']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение наивного байесовского классификатора
clf = GaussianNB()
clf.fit(X_train, y_train)

# Оценка качества классификации на обучающей выборке
train_score = clf.score(X_train, y_train)
print(f"Точность на обучающей выборке: {train_score:.2f}")

# Подсчет числа и процента неверных классификаций
y_train_pred = clf.predict(X_train)
num_errors = (y_train != y_train_pred).sum()
error_rate = num_errors / len(y_train) * 100
print(f"Число неверных классификаций: {num_errors}")
print(f"Процент неверных классификаций: {error_rate:.2f}%")

# Визуализация данных
plt.figure(figsize=(12, 8))

# Создаем уникальные маркеры и цвета для каждого класса
unique_classes = y_train.unique()
markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '8', 'P', 'X']
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))

# Отображаем точки для каждого класса
for i, (class_name, marker, color) in enumerate(zip(unique_classes, markers, colors)):
    mask = y_train == class_name
    plt.scatter(X_train.loc[mask, 'sugarpercent'], 
               X_train.loc[mask, 'pricepercent'],
               c=[color], 
               marker=marker,
               label=class_name,
               alpha=0.6)

plt.xlabel('Sugar Percent')
plt.ylabel('Price Percent')
plt.title('Распределение конфет по содержанию сахара и цене')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Дополнительная визуализация с выделением ошибок классификации
plt.figure(figsize=(12, 8))

# Правильно классифицированные объекты
correct_mask = y_train == y_train_pred
plt.scatter(X_train.loc[correct_mask, 'sugarpercent'],
           X_train.loc[correct_mask, 'pricepercent'],
           c='green', marker='o', label='Правильная классификация',
           alpha=0.6)

# Неправильно классифицированные объекты
incorrect_mask = y_train != y_train_pred
plt.scatter(X_train.loc[incorrect_mask, 'sugarpercent'],
           X_train.loc[incorrect_mask, 'pricepercent'],
           c='red', marker='x', label='Ошибка классификации',
           alpha=0.6)

plt.xlabel('Sugar Percent')
plt.ylabel('Price Percent')
plt.title('Визуализация ошибок классификации')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
