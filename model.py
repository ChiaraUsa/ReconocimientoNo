from sklearn import svm, metrics
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import numpy as np

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
        dump(scaler, 'scaler.joblib')
    return X_train, y_train, X_test, y_test

train_data, train_labels, test_data, test_labels = fetch_data()

# PCA manual
def compute_covariance_matrix(X):
    X_centered = X - np.mean(X, axis=0)  # Centrar los datos
    m = X_centered.shape[0]
    return (1 / m) * np.dot(X_centered.T, X_centered)


def pca(X, variance_threshold=0.95):
    covariance_matrix = compute_covariance_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    d = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    selected_eigenvectors = eigenvectors[:, :d]
    return selected_eigenvectors

# Calcular PCA
selected_eigenvectors = pca(train_data)
train_data_pca = np.dot(train_data, selected_eigenvectors)
test_data_pca = np.dot(test_data, selected_eigenvectors)

# Normalizar los datos reducidos
scaler = StandardScaler()
train_data_pca = scaler.fit_transform(train_data_pca)
test_data_pca = scaler.transform(test_data_pca)
dump(scaler, 'scaler_pca.joblib')

# Entrenar el modelo SVM
svc = svm.SVC(gamma='scale', class_weight='balanced', C=100)
svc.fit(train_data_pca, train_labels)

# Guardar el modelo entrenado y los componentes PCA
dump(svc, 'svm_digit_classifier_pca.joblib')
np.save('selected_eigenvectors.npy', selected_eigenvectors)

# Realizar predicciones en el conjunto de prueba
predicted = svc.predict(test_data_pca)

# Imprimir el reporte de clasificación
print(f"Classification report for classifier {svc}:\n"
      f"{metrics.classification_report(test_labels, predicted)}\n")
