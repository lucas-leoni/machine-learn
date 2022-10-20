import pandas as pd

dataIris = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)  # sep="" personalizar o separador

print(dataIris.head())
