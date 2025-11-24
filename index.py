import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from sklearn.naive_bayes import BernoulliNB

data = load_diabetes()

# ПРАВИЛЬНО: берем ДВА признака для классификации
X = data.data[:, :2]  # Первые два признака (двумерный массив!)
y = data.target

# Преобразуем target в классы (низкий/высокий)
y_class = np.where(y > np.median(y), 1, 0)

# 1. Сначала построим диаграмму рассеивания
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_class, cmap='autumn', alpha=0.7)
plt.colorbar(label='Класс (0-низкий, 1-высокий)')
plt.title('Диаграмма рассеивания Diabetes')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.show()

# 2. Теперь GaussianNB
gaussian_nb = GaussianNB()
gaussian_nb.fit(X, y_class)  # Обучаем на классах
print(gaussian_nb)

# 3. Визуализация границ классификации
X0 = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
X1 = np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100)
X0_grid, X1_grid = np.meshgrid(X0, X1)
XX = np.array([X0_grid.ravel(), X1_grid.ravel()]).T

Z = gaussian_nb.predict(XX).reshape(X0_grid.shape)

plt.figure(figsize=(10, 6))
plt.contourf(X0_grid, X1_grid, Z, alpha=0.3, cmap='autumn')
plt.scatter(X[:, 0], X[:, 1], c=y_class, cmap='autumn')
plt.title('Границы классификации GaussianNB')
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.show()
y_pred = gaussian_nb.predict(X)
print(confusion_matrix(y,y_pred))
print('Accuracy', accuracy_score(y,y_pred))
print('f1 score', f1_score(y,y_pred))

bern_nb = BernoulliNB().fit(X,y)
y_pred = bern_nb.predict(X)
print(confusion_matrix(y,y_pred))
print('Accuracy', accuracy_score(y,y_pred))
print('f1 score', f1_score(y,y_pred))
# В этом проекте по нулям все
