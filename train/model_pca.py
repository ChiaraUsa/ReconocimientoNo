# Importación de Modulos
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

def fetch_data(test_size=10000, randomize=False, standardize=True):
    """
    Obtiene el conjunto de datos MNIST desde OpenML, opcionalmente lo randomiza, lo divide en conjuntos de entrenamiento y prueba,
    y estandariza las características si se especifica.

    Parámetros:
    -----------
    test_size : int, opcional, por defecto=10000
        El número de muestras a incluir en el conjunto de prueba. El conjunto de entrenamiento contendrá las muestras restantes.

    randomize : bool, opcional, por defecto=False
        Si es True, randomiza el orden de las muestras antes de dividir en conjuntos de entrenamiento y prueba.

    standardize : bool, opcional, por defecto=True
        Si es True, estandariza las características eliminando la media y escalando a varianza unitaria.

    Retorna:
    --------
    X_train : array-like, shape (n_train_samples, n_features)
        Las muestras de entrada para el entrenamiento.

    y_train : array-like, shape (n_train_samples,)
        Los valores objetivo para el entrenamiento.

    X_test : array-like, shape (n_test_samples, n_features)
        Las muestras de entrada para la prueba.

    y_test : array-like, shape (n_test_samples,)
        Los valores objetivo para la prueba.
    """
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    if randomize:
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False)
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        dump(scaler, '../models/scaler.joblib')
    return X_train, y_train, X_test, y_test

def compute_covariance_matrix(X):
    """
    Calcula la matriz de covarianza para los datos de entrada.

    Parámetros:
    -----------
    X : array-like, shape (n_samples, n_features)
        Los datos de entrada centrados.

    Retorna:
    --------
    covariance_matrix : array-like, shape (n_features, n_features)
        La matriz de covarianza de los datos de entrada.
    """
    X_centered = X - np.mean(X, axis=0)  # Centrar los datos
    m = X_centered.shape[0]
    return (1 / m) * np.dot(X_centered.T, X_centered)

def pca(X, variance_threshold=0.95):
    """
    Aplica Análisis de Componentes Principales (PCA) para reducir la dimensionalidad de los datos.

    Parámetros:
    -----------
    X : array-like, shape (n_samples, n_features)
        Los datos de entrada.

    variance_threshold : float, opcional, por defecto=0.95
        El umbral de varianza acumulada para seleccionar el número de componentes principales.

    Retorna:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Los datos transformados a los componentes principales seleccionados.

    eigenvector_subset : array-like, shape (n_features, n_components)
        Los vectores propios correspondientes a los componentes seleccionados.
    """
    X_centered = X - np.mean(X, axis=0)  # Centrar los datos
    covariance_matrix = compute_covariance_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    explained_variances = sorted_eigenvalues / np.sum(sorted_eigenvalues)
    cumulative_explained_variance = np.cumsum(explained_variances)
    n_components = np.argmax(cumulative_explained_variance >= variance_threshold) + 1
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    X_reduced = np.dot(eigenvector_subset.transpose(), X_centered.transpose()).transpose()
    
    return X_reduced, eigenvector_subset

#Uso de la función fetch_data
train_data, train_labels, test_data, test_labels = fetch_data()

# Aplicar PCA y reducir la dimensionalidad
train_data_pca, pca_eigenvectors = pca(train_data, variance_threshold=0.95)
test_data_pca = np.dot(test_data - np.mean(train_data, axis=0), pca_eigenvectors)

# Entrenar el modelo SVM
classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_data_pca, train_labels)

# Guardar el modelo entrenado y los componentes PCA
dump(classifier, '../models/svc_digit_classifier_pca.joblib')
np.save('../models/pca_eigenvectors.npy', pca_eigenvectors)

# Realizar predicciones en el conjunto de prueba
predicted = classifier.predict(test_data_pca)

# Imprimir el reporte de clasificación
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, predicted)}\n")
