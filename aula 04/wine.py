from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

acuracias = []
for x in range(0, 100):
    dataWine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)  # sep="" personalizar o separador

    # print(dataWine.head())

    labels = dataWine[0]
    data = dataWine.drop(0, axis=1)

    # print(labels.head())
    # print(data.head())

    data_train, data_test, label_train, label_test = train_test_split(
        data, labels, test_size=0.2)
    # print("\ndata_train:\n")
    # print(data_train.head())
    # print(data_train.shape)

    # print("\ndata_test:\n")
    # print(data_test.head())
    # print(data_test.shape)

    modelo = tree.DecisionTreeClassifier()
    modelo.fit(data_train, label_train)
    predicao = modelo.predict(data_test)
    print('Predição: ', predicao)
    acuracia = accuracy_score(label_test, predicao)*100
    acuracias.append(acuracia)
    print('Acurácia: ', acuracia, '\n\n')

print('Média: ', np.mean(acuracias))  # média
print('Desvio padrão: ', np.std(acuracias))  # desvio padrão
print('Valor mínimo: ', np.min(acuracias))  # valor mínimo
print('Valor máximo: ', np.max(acuracias))  # valor máximo

predicao1 = [[13.71,1.86,2.36,16.6,101,2.61,2.88,.27,1.69,3.8,1.11,4,1035]] #1
predicao2 = [[12,1.51,2.42,22,86,1.45,1.25,.5,1.63,3.6,1.05,2.65,450]] #2
predicao3 = [[13.17,5.19,2.32,22,93,1.74,.63,.61,1.55,7.9,.6,1.48,725]] #3

print('Predição1: ', modelo.predict(predicao1))
print('Predição2: ', modelo.predict(predicao2))
print('Predição3: ', modelo.predict(predicao3))
print('\nMatriz confusão:\n')
print(confusion_matrix(predicao, label_test))