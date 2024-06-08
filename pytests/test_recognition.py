# Importación de Módulos
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn import svm, metrics
from train.model_pca import fetch_data, pca,compute_covariance_matrix
import pytest

def test_compute_covariance_matrix():
    """
    Prueba la función compute_covariance_matrix.

    Esta prueba verifica que la función compute_covariance_matrix calcule correctamente la matriz
    de covarianza para un conjunto de datos generado aleatoriamente. La función verifica dos
    condiciones:
    1. Que la forma de la matriz de covarianza es correcta.
    2. Que los valores en la matriz de covarianza calculada coinciden con los valores esperados
       usando la función np.cov.

    Pasos de la prueba:
    -------------------
    1. Generar un conjunto de datos aleatorio con dimensiones 100 x 50.
    2. Calcular la matriz de covarianza utilizando la función compute_covariance_matrix.
    3. Verificar que la matriz de covarianza tiene la forma correcta (50 x 50).
    4. Verificar que los valores en la matriz de covarianza calculada son equivalentes a los valores
       calculados usando np.cov con los mismos datos, ignorando pequeñas diferencias numéricas.

    Excepciones:
    ------------
    AssertionError: Si alguna de las condiciones de la prueba no se cumple.

    """
    data = np.random.rand(100, 50)
    covariance_matrix = compute_covariance_matrix(data)
    
    assert covariance_matrix.shape == (50, 50), "La forma de la matriz de covarianza no es correcta"
    assert np.allclose(covariance_matrix, np.cov(data, rowvar=False, bias=True)), "La matriz de covarianza no es correcta"

def test_pca():
    """
    Prueba la función pca.

    Esta prueba verifica que la función pca reduzca correctamente la dimensionalidad de un conjunto
    de datos generado aleatoriamente y que los componentes principales seleccionados expliquen al
    menos el 95% de la varianza total.

    Pasos de la prueba:
    -------------------
    1. Generar un conjunto de datos aleatorio con dimensiones 100 x 50.
    2. Aplicar PCA al conjunto de datos con un umbral de varianza acumulada del 95%.
    3. Verificar que el número de componentes seleccionados no excede el número de características
       originales (50).
    4. Calcular la matriz de covarianza del conjunto de datos.
    5. Calcular los valores propios y vectores propios de la matriz de covarianza.
    6. Ordenar los valores propios en orden descendente.
    7. Calcular la proporción de varianza explicada por cada componente principal.
    8. Calcular la varianza acumulada y determinar el número de componentes necesarios para
       alcanzar al menos el 95% de la varianza.
    9. Verificar que el número de componentes seleccionados por la función pca coincide con el
       número calculado.

    Excepciones:
    ------------
    AssertionError: Si alguna de las condiciones de la prueba no se cumple.

    """
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
    """
    Prueba de integración completa para el flujo de trabajo de PCA y SVM.

    Esta prueba verifica la correcta integración de las funciones de preprocesamiento de datos,
    reducción de dimensionalidad mediante PCA, escalado de características y entrenamiento de un
    modelo SVM. La prueba asegura que el modelo entrenado tenga una precisión aceptable en el
    conjunto de datos de prueba.

    Pasos de la prueba:
    -------------------
    1. Obtener los datos de entrenamiento y prueba utilizando la función fetch_data.
    2. Aplicar PCA al conjunto de datos de entrenamiento para reducir la dimensionalidad.
    3. Transformar el conjunto de datos de prueba utilizando los componentes principales obtenidos
       del conjunto de entrenamiento.
    4. Normalizar los datos reducidos utilizando StandardScaler.
    5. Entrenar un modelo SVM con los datos de entrenamiento reducidos y normalizados.
    6. Realizar predicciones en el conjunto de datos de prueba.
    7. Verificar que la precisión del modelo en el conjunto de datos de prueba sea superior al 90%.
    8. Imprimir el informe de clasificación del modelo.

    Excepciones:
    ------------
    AssertionError: Si la precisión del modelo es inferior al 90%.

    """
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

# Ejecución de las Pruebas
if __name__ == "__main__":
    test_compute_covariance_matrix()
    test_pca()
    test_integration()
    print("All tests passed!")