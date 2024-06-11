# Importaciones necesarias
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from joblib import dump

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

    Notas:
    ------
    - Esta función utiliza la función `fetch_openml` de `sklearn.datasets` para obtener el conjunto de datos MNIST.
    - Si `randomize` es True, las muestras se barajan usando un estado aleatorio fijo para reproducibilidad.
    - La función `train_test_split` de `sklearn.model_selection` se utiliza para dividir los datos.
    - Si `standardize` es True, se utiliza `StandardScaler` de `sklearn.preprocessing` para estandarizar las características.
    - El escalador ajustado se guarda en un archivo llamado 'scaler.joblib' en el directorio '../models/'.
    """
    # Descargar el conjunto de datos MNIST desde OpenML
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    if randomize:
        # Si randomize es True, barajar las muestras
        random_state = check_random_state(0)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        y = y[permutation]
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    if standardize:
        # Si standardize es True, estandarizar las características
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Guardar el escalador ajustado
        dump(scaler, '../models/scaler_opt.joblib')
    return X_train, y_train, X_test, y_test

# Uso de la función fetch_data para obtener los datos
train_data, train_labels, test_data, test_labels = fetch_data()

print('Inicio de data augmentation')
# Aumento de datos (Data Augmentation) utilizando ImageDataGenerator de Keras
datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,      # Rango de rotación aleatoria
    zoom_range=0.1,         # Rango de zoom aleatorio
    width_shift_range=0.1,  # Rango de desplazamiento horizontal aleatorio
    height_shift_range=0.1  # Rango de desplazamiento vertical aleatorio
)

# Aplicar el aumento de datos al conjunto de entrenamiento
train_data_augmented = train_data.reshape(-1, 28, 28, 1)  # Reformatear los datos para el generador de imágenes
datagen.fit(train_data_augmented)  # Ajustar el generador de imágenes a los datos de entrenamiento

# Generar datos aumentados
augmented_images = []
augmented_labels = []

# Generar un lote de datos aumentados
for i, (X_batch, y_batch) in enumerate(datagen.flow(train_data_augmented, train_labels, batch_size=60000)):
    if i > 0:  # Solo queremos un lote de datos aumentados
        break
    augmented_images.append(X_batch)
    augmented_labels.append(y_batch)

# Concatenar los datos aumentados en matrices numpy
augmented_images = np.concatenate(augmented_images, axis=0)
augmented_labels = np.concatenate(augmented_labels, axis=0)

# Volver a aplanar las imágenes aumentadas
augmented_images = augmented_images.reshape(-1, 784)

# Combinar los datos originales con los datos aumentados
combined_train_data = np.vstack((train_data, augmented_images))
combined_train_labels = np.hstack((train_labels, augmented_labels))

print('Inicio de PCA')

# Aplicar PCA y reducir la dimensionalidad automáticamente manteniendo el 95% de la varianza
pca = PCA(n_components=0.95)
combined_train_data_pca = pca.fit_transform(combined_train_data)
test_data_pca = pca.transform(test_data)

# Guardar el modelo PCA ajustado
dump(pca, '../models/pca_model_auto.joblib')    

print('Inicio de entrenamiento de SVC')

# Entrenar el modelo SVM
classifier = SVC(kernel="linear", random_state=6)
classifier.fit(combined_train_data_pca, combined_train_labels)

# Guardar el modelo entrenado y los componentes PCA
dump(classifier, '../models/svc_optimize_pca.joblib')

print('Inicio de predicción')

# Realizar predicciones en el conjunto de prueba
predictions = classifier.predict(test_data_pca)

# Imprimir el reporte de clasificación
print(f"Classification report for the best classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, predictions)}\n")

# Calcular la matriz de confusión
cm = metrics.confusion_matrix(test_labels, predictions)
# Visualizar la matriz de confusión
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
disp.plot()
plt.savefig('../graphs/confusion_matrix_optimize.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura

n_classes = 10  # Número de clases en MNIST
# Convertir las etiquetas a formato binarizado
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
plt.savefig('../graphs/optimize_curve_pca.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura

# Obtener los coeficientes del hiperplano
w = classifier.coef_[0]
slope = -w[0] / w[1]
intercept = -classifier.intercept_[0] / w[1]

# Crear una línea para el hiperplano de decisión
xx = np.linspace(min(combined_train_data_pca[:, 0]), max(combined_train_data_pca[:, 0]))
yy = slope * xx + intercept

# Crear márgenes (paralelos al hiperplano de decisión)
margin = 1 / np.sqrt(np.sum(classifier.coef_ ** 2))
yy_down = yy - np.sqrt(1 + slope ** 2) * margin
yy_up = yy + np.sqrt(1 + slope ** 2) * margin

# Convertir las etiquetas de clase a valores numéricos
unique_labels = np.unique(combined_train_labels)
label_to_num = {label: num for num, label in enumerate(unique_labels)}
numeric_labels = np.vectorize(label_to_num.get)(combined_train_labels)

# Plotear los datos y los hiperplanos
plt.scatter(combined_train_data_pca[:, 0], combined_train_data_pca[:, 1], c=numeric_labels, cmap='coolwarm', s=30)
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

# Resaltar los vectores de soporte
plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM con Kernel Lineal')
plt.savefig('../graphs/svc_optimize.png')  # Guardar la gráfica
plt.close()  # Cerrar la figura