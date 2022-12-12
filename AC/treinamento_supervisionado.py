from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

""" dataIris = pd.read_csv(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, encoding='utf-8')

conjunto_treinamento = dataIris.sample(100)
dados_treinamento = np.array(conjunto_treinamento)[:, :4]
labels_treinamento = np.array(conjunto_treinamento.get(4))

modelo = MultinomialNB()
modelo.fit(dados_treinamento, labels_treinamento)

teste = [[6.7, 3.3, 5.7, 2.5],] # Iris-virginica

print(modelo.predict(teste)) """

acuracias = []
for x in range(0, 4):
  dataIris = pd.read_csv(
      "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
  # print(dataIris.head())
  conjunto_treinamento = dataIris.sample(100)
  labels = np.array(conjunto_treinamento.get(4))
  data = np.array(conjunto_treinamento)[:, :4]

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
  modelo = MultinomialNB()
  modelo.fit(data_train, label_train)
  predicao = modelo.predict(data_test)
  print('Predição:\n', predicao)
  acuracia = accuracy_score(label_test, predicao)*100
  acuracias.append(acuracia)
  print('Acurácia: ', acuracia, '\n')

print('Média: ', np.mean(acuracias))  # média
print('Desvio padrão: ', np.std(acuracias))  # desvio padrão
print('Valor mínimo: ', np.min(acuracias))  # valor mínimo
print('Valor máximo: ', np.max(acuracias))  # valor máximo

dados1 = [[5.1,3.5,1.4,0.2]] #Iris-setosa
dados2 = [[7.0,3.2,4.7,1.4]] #Iris-versicolor
dados3 = [[6.3,3.3,6.0,2.5]] #Iris-virginica

predicao1 = modelo.predict(dados1)
predicao2 = modelo.predict(dados2)
predicao3 = modelo.predict(dados3)

print('Predição Iris-Setosa: ', predicao1)
print('Predição Iris-Versicolor: ', predicao2)
print('Predição Iris-Virginica: ', predicao3)