from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

acuracias = []
for x in range(0, 100):
    dataWine = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/reprocessed.hungarian.data", header=None)  # sep="" personalizar o separador

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
    predict = modelo.predict(data_test)
    print('Predict: ', predict)
    acuracia = accuracy_score(label_test, predict)*100
    acuracias.append(acuracia)
    print('Acurácia: ', acuracia, '\n\n')

print('Média: ', np.mean(acuracias))  # média
print('Desvio padrão: ', np.std(acuracias))  # desvio padrão
print('Valor mínimo: ', np.min(acuracias))  # valor mínimo
print('Valor máximo: ', np.max(acuracias))  # valor máximo