from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

def fetch_data(test_size=10000, randomize=False, standardize=True):
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
    X_centered = X - np.mean(X, axis=0)  # Centrar los datos
    m = X_centered.shape[0]
    return (1 / m) * np.dot(X_centered.T, X_centered)

def pca(X, variance_threshold=0.95):
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

train_data, train_labels, test_data, test_labels = fetch_data()

# Aplicar PCA y reducir la dimensionalidad
train_data_pca, pca_eigenvectors = pca(train_data, variance_threshold=0.95)
test_data_pca = np.dot(test_data - np.mean(train_data, axis=0), pca_eigenvectors)

# Entrenar el modelo SVM
classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_data_pca, train_labels)

# Guardar el modelo entrenado y los componentes PCA
dump(classifier, '../models/svc_digit_classifier_pca.joblib')
dump(pca_eigenvectors, "../models/pca_eigenvectors.npy")

# Realizar predicciones en el conjunto de prueba
predicted = classifier.predict(test_data_pca)

# Imprimir el reporte de clasificación
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, predicted)}\n")
