# Importaciones necesarias
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

#Función para obtener los datos
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

#Uso de la función fetch_data
train_data, train_labels, test_data, test_labels = fetch_data()

# Crear una instancia del clasificador SVM con un kernel lineal y un estado aleatorio específico.
classifier=SVC(kernel="linear", random_state=6)

# Entrenar el clasificador utilizando los datos y etiquetas de entrenamiento.
classifier.fit(train_data,train_labels)

# Guardar el modelo entrenado en un archivo para uso futuro.
dump(classifier, "../models/digit_recognizer")

# Utilizar el clasificador entrenado para predecir las etiquetas del conjunto de datos de prueba.
prediction=classifier.predict(test_data)

# Imprimir un informe de clasificación que incluye métricas como precisión, recall y f1-score.
print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, prediction)}\n")        