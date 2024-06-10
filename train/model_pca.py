# Importación de Modulos
from sklearn import metrics
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler, label_binarize
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
        dump(scaler, '../models/scaler_pca.joblib')
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

# Calcular la matriz de confusión
cm = metrics.confusion_matrix(test_labels, predicted)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.savefig('../graphs/confusion_matrix_pca.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura

n_classes = 10  # Número de clases en MNIST
y_test_binarized = label_binarize(test_labels, classes=[str(i) for i in range(n_classes)])

# Calcular la función de decisión para cada clase
decision_function = classifier.decision_function(test_data_pca)

# Calcular y trazar la curva Precision-Recall para cada clase
for i in range(n_classes):
    precision, recall, _ = metrics.precision_recall_curve(y_test_binarized[:, i], decision_function[:, i])
    plt.plot(recall, precision, lw=2, label=f'Class {i}')
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall por Clase')
plt.legend(loc='best')
plt.savefig('../graphs/precision_recall_curve_pca.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura

# Obtener los coeficientes del hiperplano
w = classifier.coef_[0]
slope = -w[0] / w[1]
intercept = -classifier.intercept_[0] / w[1]

# Crear una línea para el hiperplano de decisión
xx = np.linspace(min(train_data_pca[:, 0]), max(train_data_pca[:, 0]))
yy = slope * xx + intercept

# Crear márgenes (paralelos al hiperplano de decisión)
margin = 1 / np.sqrt(np.sum(classifier.coef_ ** 2))
yy_down = yy - np.sqrt(1 + slope ** 2) * margin
yy_up = yy + np.sqrt(1 + slope ** 2) * margin

# Convertir las etiquetas de clase a valores numéricos
unique_labels = train_labels.unique()
label_to_num = {label: num for num, label in enumerate(unique_labels)}
numeric_labels = train_labels.map(label_to_num)

# Plotear los datos y los hiperplanos
plt.scatter(train_data_pca[:, 0], train_data_pca[:, 1], c=numeric_labels, cmap='coolwarm', s=30)
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Resaltar los vectores de soporte
plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM con Kernel Lineal')
plt.savefig('../graphs/svc_pca.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura