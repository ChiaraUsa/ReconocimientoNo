import cv2
import numpy as np
from joblib import load
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Cargar el modelo SVM, el scaler y el modelo PCA
clf = load('../models/svm_digit_classifier_pca.joblib')
scaler = load('../models/scaler.joblib')
pca_model = load('../models/pca.joblib')

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
        if w > 5 and h > 5:
            digit = image[y:y+h, x:x+w]
            digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            digit = digit.astype('float32') / 255
            cv2.imshow('frame2', digit)
            digit = digit.reshape(1, -1)  # Ajustar a 2D
            digits.append(digit)
            positions.append((x, y, w, h))
    return digits, positions

def recognize_digits(digits, scaler, pca_model):
    predictions = []
    if digits:
        digits_array = np.array(digits).reshape(len(digits), -1)
        standardized_digits = scaler.transform(digits_array)
        projected_digits = pca_model.transform(standardized_digits)
        predictions = clf.predict(projected_digits)
    return predictions

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Obtener el tamaño del frame
ret, frame = cap.read()
height, width, _ = frame.shape

# Definir la región de interés (ROI)
roi_size = 200
roi_top_left = ((width - roi_size) // 2, (height - roi_size) // 2)
roi_bottom_right = ((width + roi_size) // 2, (height + roi_size) // 2)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo recibir el frame. Saliendo...")
        break

    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)
    roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
    processed_image = preprocess_image(roi)
    digits, positions = segment_digits(processed_image)
    
    if digits:
        recognized_digits = recognize_digits(digits, scaler, pca_model)

        for idx, (digit, (x, y, w, h)) in enumerate(zip(recognized_digits, positions)):
            cv2.putText(frame, f'{digit}', (x + roi_top_left[0], y + roi_top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
