# Импортируем библиотеки
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Импортируем из библиотеки скилерн  датасет 
from sklearn.datasets import load_breast_cancer
# Загружаем данные 
data = load_breast_cancer()
# Выбираем из X только 2 столбца
X = data.data
y = data.target;
print(X)
print(y)

plt.scatter(X[:,0],X[:,1], c=y,cmap='autumn')
plt.title('Визуализация данных')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

plt.figure(figsize=(8,3))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.hist(X[:,i])
    plt.xlabel(data.feature_names[i])
    plt.suptitle('Гистограммы признаков', fontsize=(14))
plt.show()

from sklearn.naive_bayes import GaussianNB
#gaussian_nb = GaussianNB()
#gaussian_nb.fit(X,y)

#X0 = np.linspace(X[:,0].min()-1, X[:,0].max()+1, X.shape[0])
#X1 = np.linspace(X[:,1].min()-1, X[:,1].max()+1, X.shape[0])
#X0_grid,X1_grid = np.meshgrid(X0,X1)
#XX = np.array([X0_grid.ravel(),X1_grid.ravel()]).T

#Z = gaussian_nb.predict(XX).reshape(X0_grid.shape)
#plt.scatter(X[:,0],X[:,1],c=y)
#plt.contourf(X0_grid, X1_grid, Z, alpha=0.2)

#plt.xlabel('mean radius')
#plt.ylabel('mean texture')
#plt.title('Диаграмма рассеивания с областями классификации')
#plt.show()
#y_pred = gaussian_nb.predict(X)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
#print(confusion_matrix(y,y_pred))
#print('Accuracy', accuracy_score(y,y_pred))
#print('f1 score', f1_score(y,y_pred))
# Мульмодальный
#from sklearn.naive_bayes import MultinomialNB
#miltinom_nb = MultinomialNB().fit(X,y)
#y_pred = miltinom_nb.predict(X)
#print(confusion_matrix(y,y_pred))
#print('Accuracy', accuracy_score(y,y_pred))
#print('f1 score', f1_score(y,y_pred))
# Метод Бернулли
from sklearn.naive_bayes import BernoulliNB
#bern_nb = BernoulliNB().fit(X,y)
#y_pred = bern_nb.predict(X)
#print(confusion_matrix(y,y_pred))
#print('Accuracy', accuracy_score(y,y_pred))
#print('f1 score', f1_score(y,y_pred))

f = plt.figure(figsize=(15,7))
for i in range(30):
    plt.subplot(6, 5, i + 1)
    plt.hist(X[:,i])
    plt.xlabel(data.feature_names[i])
    
    
f.subplots_adjust(hspace=0.9, wspace=0.3)
plt.suptitle('Гистограмма признаков', fontsize=14)
plt.show()

df = pd.DataFrame(data.data, columns=data.feature_names)
df = df.drop(['mean concavity', 'radius error', 'perimeter error', 'area error', 
              'compactness error', 'concavity error', 'fractal dimension error', 
              'worst concavity'], axis=1)
print(df.head())
X = df
y = data.target
print(X.shape, y.shape)
gaussian_nb = GaussianNB().fit(X,y)
y_pred = gaussian_nb.predict(X)
print(confusion_matrix(y, y_pred))
print('Accuracy', accuracy_score(y,y_pred))
print('f1 score', f1_score(y,y_pred))