import cv2
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eig

# Cargar el modelo SVM
clf = load('svm_digit_classifier.joblib')

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

def segment_digits(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits = []
    positions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filtrar contornos pequeños para reducir ruido
        if w > 5 and h > 5:
            digit = image[y:y+h, x:x+w]
            digit = cv2.resize(digit, (8, 8), interpolation=cv2.INTER_AREA)
            digit = digit.astype('float32') / 255
            digit = digit.reshape(-1)
            digits.append(digit)
            positions.append((x, y, w, h))
    return digits, positions

def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data, scaler

def compute_covariance_matrix(data):
    return np.cov(data, rowvar=False)

def compute_eigenvectors_and_values(cov_matrix):
    eigenvalues, eigenvectors = eig(cov_matrix)
    return eigenvalues, eigenvectors

def project_data(data, eigenvectors):
    return data @ eigenvectors.T

def recognize_digits(digits, scaler, eigenvectors):
    predictions = []
    if digits:
        standardized_digits = scaler.transform(digits)
        projected_digits = project_data(standardized_digits, eigenvectors)
        predictions = clf.predict(projected_digits)
    return predictions

# Inicializar la captura de video desde la cámara (0 es usualmente la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Obtener el tamaño del frame
ret, frame = cap.read()
height, width, _ = frame.shape

# Definir la región de interés (ROI) pequeña y centrada
roi_size = 100  # Tamaño del ROI
roi_top_left = ((width - roi_size) // 2, (height - roi_size) // 2)  # Coordenadas de la esquina superior izquierda
roi_bottom_right = ((width + roi_size) // 2, (height + roi_size) // 2)  # Coordenadas de la esquina inferior derecha

# Asumiendo que tienes un conjunto de datos para ajustar el PCA
example_data = np.random.rand(1000, 64)  # Datos de ejemplo para ajustar PCA
standardized_example_data, scaler = standardize_data(example_data)
cov_matrix = compute_covariance_matrix(standardized_example_data)
eigenvalues, eigenvectors = compute_eigenvectors_and_values(cov_matrix)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir el frame (stream end?). Saliendo...")
        break

    # Dibujar el cuadrado en el frame original para visualizar la ROI
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

    # Extraer la región de interés
    roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

    processed_image = preprocess_image(roi)
    digits, positions = segment_digits(processed_image)
    recognized_digits = recognize_digits(digits, scaler, eigenvectors)

    # Dibujar los dígitos reconocidos en la ROI
    for (x, y, w, h), digit in zip(positions, recognized_digits):
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(roi, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el frame en una ventana llamada 'frame'
    cv2.imshow('frame', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
