from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np

dataIris = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)  # sep="" personalizar o separador

""" setosa = dataIris[:50]
versicolor = dataIris[51:100]
virginica = dataIris[101:150]
print('\nSetosa\n\n', setosa, '\n\nVersicolor\n\n', versicolor, '\n\nVirginica\n\n', virginica) """

conjunto_treinamento = dataIris.sample(60)
labels_treinamento = np.array(conjunto_treinamento.get(4))
dados_treinamento = np.array(conjunto_treinamento)[:, :4]

""" print(dados_treinamento)
print(labels_treinamento) """

modelo = MultinomialNB()
modelo.fit(dados_treinamento, labels_treinamento)

teste = [[6.7, 3.3, 5.7, 2.5],  # Iris-virginica
         ]
print(modelo.predict(teste))
