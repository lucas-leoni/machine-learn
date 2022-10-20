"""
Aprendizado supervisionado com Multinomial Naive Bayes
"""

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()

"""
    Rosa = 1
    Margarida = -1
    cor, sepala, caule, espinhos
    cor:    1 rosa      0 branco
    sepala: 1 pontuda   0 arredondado
    caule:  1 longo    0 curto
"""
rosa1 = [1, 1, 1, 1]
rosa2 = [0, 1, 1, 1]
rosa3 = [0, 1, 0, 1]
margarida1 = [0, 0, 0, 0]
margarida2 = [1, 1, 0, 0]
margarida3 = [1, 0, 1, 0]

dados_treinamento = [rosa1, rosa2, rosa3, margarida1, margarida2, margarida3]
labels = ["rosa", "rosa", "rosa", "margarida", "margarida", "margarida"]

modelo.fit(dados_treinamento, labels)


teste = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]]
print(modelo.predict(teste))
