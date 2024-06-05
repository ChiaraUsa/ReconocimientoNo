from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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

train_data, train_labels, test_data, test_labels = fetch_data()

classifier=SVC(kernel="linear", random_state=6)
classifier.fit(train_data,train_labels)
dump(classifier, "../models/digit_recognizer")

prediction=classifier.predict(test_data)

print(f"Classification report for classifier {classifier}:\n"
      f"{metrics.classification_report(test_labels, prediction)}\n")        