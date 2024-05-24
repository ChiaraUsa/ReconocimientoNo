import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump

# Cargar el conjunto de datos MNIST
digits = datasets.load_digits()

# Preprocesar los datos
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo SVM
clf = SVC(gamma=0.001)
clf.fit(X_train, y_train)

# Evaluar el modelo
print(f"Accuracy: {clf.score(X_test, y_test) * 100:.2f}%")

# Guardar el modelo y el escalador
dump(clf, 'svm_digit_classifier.joblib')
dump(scaler, 'scaler.joblib')
