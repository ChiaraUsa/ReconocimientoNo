import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Desactivar los mensajes de información de TensorFlow
tf.get_logger().setLevel('ERROR')

# Cargar el modelo preentrenado en el conjunto de datos MNIST
model = load_model('mnist_model.h5')

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
            digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
            digit = digit.astype('float32') / 255
            digit = np.expand_dims(digit, axis=-1)
            digits.append(digit)
            positions.append((x, y, w, h))
    return digits, positions

def recognize_digits(digits):
    predictions = []
    for digit in digits:
        digit = np.expand_dims(digit, axis=0)
        prediction = model.predict(digit)
        predicted_digit = np.argmax(prediction)
        predictions.append(predicted_digit)
    return predictions

# Inicializar la captura de video desde la cámara (0 es usualmente la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# Definir la región de interés (ROI)
roi_top_left = (100, 100)  # Coordenadas de la esquina superior izquierda del cuadrado
roi_bottom_right = (300, 300)  # Coordenadas de la esquina inferior derecha del cuadrado

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
    recognized_digits = recognize_digits(digits)

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
