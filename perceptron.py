from itertools import combinations
import numpy as np
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_openml
# from tensorflow.keras.datasets import mnist

#Inicio de Mpi (no implementado)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#Vectores y caracteristicas de mnist
mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"], mnist["target"]
Y = Y.astype(np.int8)

X = X.to_numpy()

#print(mnist.DESCR)

#Clasificación binaria por clases
clases = np.unique(Y)
combinaciones = list(combinations(clases, 2))

pares_filtrados = {}

for (clase1, clase2) in combinaciones:
    
    filtro = (Y == clase1) | (Y == clase2)
    
    X_filtro = X[filtro]
    Y_filtro = Y[filtro]
    
    Y_filtro[Y_filtro == clase1] = -1
    Y_filtro[Y_filtro == clase2] = 1
    
    X_filtro = np.hstack((X_filtro, np.ones((X_filtro.shape[0], 1))))
    
    pares_filtrados[(clase1, clase2)] = (X_filtro, Y_filtro)

#Normalizar datos
X_norm = np.zeros_like(X, dtype=float)
for i in range(X.shape[0]):
    X_norm[i] = X[i] / np.linalg.norm(X[i])

#División de conjuntos en entrenamiento y prueba
X_norm,Xts,Y,Yts=train_test_split(X_norm,Y,test_size=0.2,train_size=0.8)
print(f'Tamano del conjunto de entrenamiento: Y={len(Y)}')


#Número de perceptrones y número épocas para entrenamiento
perceptrones=45
epocas=5

#Inicialización de alfa 
alfas = np.random.uniform(-0.5, 0.5, (perceptrones, len(X[0])))

#Tomamos los valores de x al azar para evitar un sesgo por ordenamiento previo
x = np.random.choice(range(len(X_norm)), len(X_norm), replace=False)
print("Valores de x: ", x)
print("Tamano de x:", len(x))
Y = np.array(Y)
Y=Y[x]

#Entrenamiento del modelo
for e in range(epocas):
    print(f"Epoca {e+1}/{epocas}")
    
    # Obtener f(z)
    z = np.dot(X_norm[x], alfas.T)
    fz = np.where(z >= 0, 1, -1)
    
    # Obtenemos p
    p = np.sum(fz, axis=1)
    
    # Obtenemos o testada sumando los resultados
    sp = np.where(p >= 0, 1, -1)
    
    # Regla p-delta para conjunto de entrenamiento (actualización de alfas)
    e = 0.01
    y = 0.005
    mu = 1
    eta = 1 / (4 * np.sqrt(epocas))
    #itera sobre conjunto de datos
    for o, z_val in enumerate(x):
        #itera sobre pesos de los perceptrones
        for i, a in enumerate(alfas):
            if sp[o] > (Y[z_val] + e) and z[o][i] >= 0:
                alfas[i] += eta * (-X_norm[z_val])
            elif sp[o] < (Y[z_val] - e) and z[o][i] < 0:
                alfas[i] += eta * X_norm[z_val]
            elif sp[o] <= (Y[z_val] + e) and 0 <= z[o][i] < y:
                alfas[i] += eta * mu * X_norm[z_val]
            elif sp[o] >= (Y[z_val] - e) and -y < z[o][i] < 0:
                alfas[i] += eta * mu * (-X_norm[z_val])
            else:
                0

accuracy = accuracy_score(Y, sp)
print(f'Precision del conjunto de entrenamiento: {accuracy:.2f}')


#PRUEBA DEL CONJUNTO
# Genera una selección aleatoria de índices del conjunto de prueba
x = np.random.choice(range(len(Xts)), len(Xts), replace=False)
#Reordenar etiquetas
Yts = np.array(Yts)
Yts = Yts[x]

# Almacenar las salidas de los perceptrones y las predicciones (az: perceptron, fz: prediccion)
az = []
fz = []

# Itera sobre los índices de x
for o in x:
    # Almacenar datos de los perceptrones y predicciones
    temporal_az = []
    temporal_fz = []

    # Itera sobre los pesos de alfa y sus índices
    for i, a in enumerate(alfas):

        pp = np.dot(Xts[o], a)
        resultado = 1 if pp >= 0 else -1
        
        temporal_az.append(pp)
        temporal_fz.append(resultado)
    
    az.append(temporal_az)
    fz.append(temporal_fz)

# Suma de predicciones
p = [np.sum(f) for f in fz]

# Comparación de resultados para obtener predicción
sp = np.where(np.array(p) >= 0, 1, -1)

print("\n")
print(f'Tamano del conjunto de prueba: Y={len(Yts)}')
print(f'Precision del conjunto de prueba: ', accuracy_score(Yts, sp))

print('Informe de clasificacion:')
print(classification_report(Yts, sp, zero_division=0))