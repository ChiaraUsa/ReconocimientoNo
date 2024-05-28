import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Cargar el conjunto de datos MNIST
digits = datasets.load_digits()

# Aplanar las imágenes
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_train, X_test = data[:n_samples // 2], data[n_samples // 2:]
y_train, y_test = digits.target[:n_samples // 2], digits.target[n_samples // 2:]

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear un clasificador SVM
clf = svm.SVC(gamma=0.001)

# Entrenar el clasificador en el conjunto de entrenamiento
clf.fit(X_train, y_train)

# Guardar el modelo entrenado y el scaler
dump(clf, 'svm_digit_classifier.joblib')
dump(scaler, 'scaler.joblib')

# Realizar predicciones en el conjunto de prueba
predicted = clf.predict(X_test)

# Imprimir el reporte de clasificación
print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(y_test, predicted)}\n")
