import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import tensorflow as tf

# Cargar los datos MNIST
def load_mnist_data(randomize=False, standardize=True):
    """
    Carga el conjunto de datos MNIST, con opciones para aleatorizar y estandarizar los datos.

    Parámetros:
    randomize (bool): Si es True, aleatoriza los datos de entrenamiento y prueba.
    standardize (bool): Si es True, estandariza los datos utilizando StandardScaler.

    Retorna:
    X_train (ndarray): Imágenes de entrenamiento aplanadas y normalizadas.
    X_test (ndarray): Imágenes de prueba aplanadas y normalizadas.
    y_train (ndarray): Etiquetas de las imágenes de entrenamiento.
    y_test (ndarray): Etiquetas de las imágenes de prueba.
    """
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Aleatorizar los datos si se especifica
    if randomize:
        random_state = check_random_state(0)
        permutation_train = random_state.permutation(X_train.shape[0])  
        permutation_test = random_state.permutation(X_test.shape[0])
        X_train = X_train[permutation_train]
        y_train = y_train[permutation_train]
        X_test = X_test[permutation_test]
        y_test = y_test[permutation_test]
    
    # Aplanar las imágenes y normalizar los píxeles
    X_train = X_train.reshape(-1, 28*28).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28*28).astype('float32') / 255.0
    
    # Estandarizar los datos si se especifica
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Guardar el escalador ajustado
        joblib.dump(scaler, 'scaler.joblib')
    
    return X_train, X_test, y_train, y_test

# Llamada a la función para cargar y procesar los datos MNIST
X_train, X_test, y_train, y_test = load_mnist_data(randomize=True, standardize=True)

# Reducir las dimensiones a 2D con PCA
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Seleccionar una clase para visualización (por ejemplo, la clase 0)
class_of_interest = 0
y_train_binary = (y_train == class_of_interest).astype(int)

# Entrenar un nuevo SVM binario para la clase seleccionada
svm_binary = SVC(C=10, kernel='poly')
svm_binary.fit(X_train_reduced, y_train_binary)

# Función para visualizar el margen suave en 2D
def plot_soft_margin(clf, X, y):
    """
    Visualiza el margen suave de un clasificador SVM en un espacio 2D reducido.

    Parámetros:
    clf (SVC): El clasificador SVM entrenado.
    X (ndarray): Datos de entrada reducidos a 2D.
    y (ndarray): Etiquetas binarias para la clase de interés.

    Esta función genera y guarda un gráfico que muestra los márgenes del SVM.
    """
    print(f"Tamaño de X_reduced: {X.shape}")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    print(f"Tamaño de xx: {xx.shape}, Tamaño de yy: {yy.shape}")
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    print(f"Tamaño de grid: {grid.shape}")
    
    Z = clf.decision_function(grid)
    print(f"Tamaño de Z antes de reshaping: {Z.shape}")
    Z = Z.reshape(xx.shape)
    print(f"Tamaño de Z después de reshaping: {Z.shape}")
    
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    
    # Dibujar los márgenes suaves
    plt.contour(xx, yy, Z, levels=[-1, 1], linewidths=2, colors='green', linestyles='dashed')
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Soft Margin for Class {class_of_interest} in 2D PCA Space')
    plt.savefig('../graphs/margenes_por_clase.png')
    plt.close()

# Visualizar el margen suave en 2D
plot_soft_margin(svm_binary, X_train_reduced, y_train_binary)
