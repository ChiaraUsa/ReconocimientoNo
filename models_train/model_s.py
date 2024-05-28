import numpy as np
import pandas as pd
from numpy.linalg import eig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from joblib import dump

# Cargar los datos
X = np.load('datasets/X.npy')
Y = np.load('datasets/Y.npy')

# Normalizar las imágenes
x_scaled = X / 255
X_flattened = x_scaled.reshape(len(X), 64*64)

# Convertir Y de one-hot encoding a etiquetas
y_labels = np.argmax(Y, axis=1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y_labels, test_size=0.2, random_state=42)

# Normalización de los datos sin usar librerías
def normalize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

X_train_normalized, mean_train, std_train = normalize_data(X_train)
X_test_normalized = (X_test - mean_train) / std_train

# Calcular la matriz de covarianza
def compute_covariance_matrix(X):
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, X)

x_cov = compute_covariance_matrix(X_train_normalized)

# Calcular autovalores y autovectores
valEIG, vecEIG = eig(x_cov)

# Ordenar autovalores y autovectores
sorted_indices = np.argsort(valEIG)[::-1]
eigenvalues = valEIG[sorted_indices]
eigenvectors = vecEIG[:, sorted_indices]

# Calcular la varianza acumulada
explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
d = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components needed to explain 95% variance: {d}")

# Reducir la dimensionalidad de los datos
selected_eigenvectors = eigenvectors[:, :d]
X_train_reduced = np.dot(X_train_normalized, selected_eigenvectors)
X_test_reduced = np.dot(X_test_normalized, selected_eigenvectors)

# Inversa para visualizar las imágenes originales (opcional)
X_train_recovered = np.dot(X_train_reduced, selected_eigenvectors.T) * std_train + mean_train

# Entrenar el modelo SVC
model_svc = SVC()
model_svc.fit(X_train_reduced, y_train)

# Guardar el modelo
dump(model_svc, 'models/model_svc.joblib')
np.save('datasets/mean_train.npy', mean_train)
np.save('datasets/std_train.npy', std_train)

# Hacer predicciones y calcular la precisión
y_pred = model_svc.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))
