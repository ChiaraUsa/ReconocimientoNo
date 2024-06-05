import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics
from train.model import fetch_data, pca,compute_covariance_matrix
import pytest

def test_compute_covariance_matrix():
    data = np.random.rand(100, 50)
    covariance_matrix = compute_covariance_matrix(data)
    
    assert covariance_matrix.shape == (50, 50), "La forma de la matriz de covarianza no es correcta"
    assert np.allclose(covariance_matrix, np.cov(data, rowvar=False, bias=True)), "La matriz de covarianza no es correcta"

def test_pca():
    data = np.random.rand(100, 50)
    selected_eigenvectors = pca(data, variance_threshold=0.95)
    
    assert selected_eigenvectors.shape[1] <= 50, "El número de componentes seleccionados no es correcto"
    
    # Verificar que la varianza explicada es >= 95%
    covariance_matrix = compute_covariance_matrix(data)
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    d = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    assert selected_eigenvectors.shape[1] == d, "El número de componentes seleccionados no explica el 95% de la varianza"

def test_integration():
    train_data, train_labels, test_data, test_labels = fetch_data()

    # Calcular PCA
    selected_eigenvectors = pca(train_data)
    train_data_pca = np.dot(train_data, selected_eigenvectors)
    test_data_pca = np.dot(test_data, selected_eigenvectors)

    # Normalizar los datos reducidos
    scaler = StandardScaler()
    train_data_pca = scaler.fit_transform(train_data_pca)
    test_data_pca = scaler.transform(test_data_pca)
    
    # Entrenar el modelo SVM
    svc = svm.SVC(gamma='scale', class_weight='balanced', C=100)
    svc.fit(train_data_pca, train_labels)
    
    # Realizar predicciones en el conjunto de prueba
    predicted = svc.predict(test_data_pca)
    
    # Verificar la precisión
    accuracy = metrics.accuracy_score(test_labels, predicted)
    assert accuracy > 0.90, "La precisión del modelo es inferior al 90%"

    print(f"Classification report for classifier {svc}:\n"
          f"{metrics.classification_report(test_labels, predicted)}\n")

if __name__ == "__main__":
    test_compute_covariance_matrix()
    test_pca()
    test_integration()
    print("All tests passed!")