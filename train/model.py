from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    pca = PCA(n_components=variance_threshold, svd_solver='full')
    X_reduced = pca.fit_transform(X)
    dump(pca, '../models/pca.joblib')
    return X_reduced, pca

train_data, train_labels, test_data, test_labels = fetch_data()

# Aplicar PCA y reducir la dimensionalidad
train_data_pca, pca_model = pca(train_data)
test_data_pca = pca_model.transform(test_data)

# Entrenar el modelo SVM
classifier=SVC(kernel="linear", random_state=6)
classifier.fit(train_data_pca,train_labels)

# Guardar el modelo entrenado y los componentes PCA
dump(classifier, '../models/svc_digit_classifier_pca.joblib')

# Realizar predicciones en el conjunto de prueba
predicted = classifier.predict(test_data_pca)

# Imprimir el reporte de clasificación
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, predicted)}\n")
