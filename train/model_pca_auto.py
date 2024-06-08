# Importaciones necesarias
from sklearn import metrics, precision_recall_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Función para obtener los datos
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        dump(scaler, '../models/scaler.joblib')
    return X_train, y_train, X_test, y_test

# Uso de la función fetch_data
train_data, train_labels, test_data, test_labels = fetch_data()

# Aplicar PCA y reducir la dimensionalidad
pca = PCA(n_components=0.95)
train_data_pca = pca.fit_transform(train_data)
test_data_pca = pca.transform(test_data)

# Guardar el modelo PCA
dump(pca, '../models/pca_model_auto.joblib')

# Crear una instancia del clasificador SVM con un kernel lineal
classifier = SVC(kernel="linear", random_state=6)
classifier.fit(train_data_pca, train_labels)

# Guardar el modelo entrenado 
dump(classifier, '../models/svc_digit_classifier_pca_optimized.joblib')

# Realizar predicciones en el conjunto de prueba
predictions = classifier.predict(test_data_pca)

# Imprimir el reporte de clasificación
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, predictions)}\n")

# Calcular las probabilidades de decisión
decision_function = classifier.decision_function(test_data_pca)

# Curva de Precisión-Exhaustividad
precision, recall, _ = precision_recall_curve(test_labels, decision_function)
plt.figure()
plt.plot(recall, precision, lw=2, color='b', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()      

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

# Plotear los datos y los hiperplanos
plt.scatter(train_data_pca[:, 0], train_data_pca[:, 1], c=train_labels, cmap='coolwarm', s=30)
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Resaltar los vectores de soporte
plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM with Linear Kernel')
plt.show()
