import cv2
import numpy as np
from joblib import load
from numpy.linalg import eig

# Cargar el modelo SVM
clf = load('models/model_svc.joblib')
mean_train = np.load('datasets/mean_train.npy')
std_train = np.load('datasets/std_train.npy')

# Normalizar los datos
def normalize_data(X, mean, std):
    return (X - mean) / std

# Calcular la matriz de covarianza
def compute_covariance_matrix(X):
    m = X.shape[0]
    return (1 / m) * np.dot(X.T, X)

# Calcular autovalores y autovectores
def compute_eigenvectors_and_values(cov_matrix):
    eigenvalues, eigenvectors = eig(cov_matrix)
    return np.real(eigenvalues), np.real(eigenvectors)

# Seleccionar autovectores que explican la varianza especificada
def select_eigenvectors(eigenvalues, eigenvectors, variance_threshold=0.95):
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    d = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    return eigenvectors[:, :d]

# Proyectar los datos a un espacio de menor dimensionalidad
def project_data(data, eigenvectors):
    return np.dot(data, eigenvectors)

# Preprocesar imagen
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return thresh

# Segmentar dígitos en la imagen
def segment_digits(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digits, positions = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:
            digit = image[y:y+h, x:x+w]
            digit = cv2.resize(digit, (64, 64), interpolation=cv2.INTER_AREA)
            digit = digit.astype('float32') / 255
            digit = digit.reshape(-1)
            digits.append(digit)
            positions.append((x, y, w, h))
    return digits, positions

# Reconocer dígitos en la imagen
def recognize_digits(digits, mean, std, eigenvectors):
    predictions = []
    if digits:
        normalized_digits = normalize_data(np.array(digits), mean, std)
        projected_digits = project_data(normalized_digits, eigenvectors)
        predictions = clf.predict(projected_digits)
    return predictions

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Obtener el tamaño del frame
ret, frame = cap.read()
if not ret:
    print("No se pudo recibir el frame inicial. Saliendo...")
    cap.release()
    exit()

height, width, _ = frame.shape

# Definir la región de interés (ROI)
roi_size = 100
roi_top_left = ((width - roi_size) // 2, (height - roi_size) // 2)
roi_bottom_right = ((width + roi_size) // 2, (height + roi_size) // 2)

# Calcular la matriz de covarianza y los autovectores
example_data = np.random.rand(1000, 64*64)  # Datos de ejemplo para ajustar PCA
normalized_example_data = normalize_data(example_data, mean_train, std_train)
cov_matrix = compute_covariance_matrix(normalized_example_data)
eigenvalues, eigenvectors = compute_eigenvectors_and_values(cov_matrix)
selected_eigenvectors = select_eigenvectors(eigenvalues, eigenvectors)

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
    recognized_digits = recognize_digits(digits, mean_train, std_train, selected_eigenvectors)

    # Dibujar los dígitos reconocidos en la ROI
    for (x, y, w, h), digit in zip(positions, recognized_digits):
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(digit), (x + roi_top_left[0], y + roi_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el frame en una ventana llamada 'frame'
    cv2.imshow('frame', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
