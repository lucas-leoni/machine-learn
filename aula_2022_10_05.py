import matplotlib.pyplot as plt
import numpy as np

vetor_teste = [3, 3, 3, -5, 3, 6, 3, 3, 3]
# Calcula a média
print('Média:', np.mean(vetor_teste))
# Calcula a mediana
print('Mediana:', np.median(vetor_teste))
# Calcula a média
print('Média:', np.average(vetor_teste))
# Remove valores repetidos e ordena
print('Valores únicos:', np.unique(vetor_teste))
# Desvio padrão
print('Desvio padrão:', np.std(vetor_teste))
# Maior valor
print('Maior valor:', np.amax(vetor_teste))
# Menor valor
print('Menor valor:', np.amin(vetor_teste))
# Cria array com um determinado número de elementos
print('Criar array:', np.arange(9))
print('Criar array:', np.arange(600))

array_a = np.array([2, 6, 9])
array_b = np.array([8, 5, 20])

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# Valores menor que 10
print('Menor que 10:', a[a < 10])

# Matplotlib
plt.style.use('_mpl-gallery-nogrid')
# make data
x = [1, 2, 3, 4, 10]
colors = plt.get_cmap('Greens')(np.linspace(0.2, 0.7, len(x)))
# plot
fig, ax = plt.subplots()
ax.pie(x, colors=colors, radius=6, center=(8, 8),
       wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
ax.set(xlim=(0, 16), xticks=np.arange(1, 16),
       ylim=(0, 16), yticks=np.arange(1, 16))
plt.show()
