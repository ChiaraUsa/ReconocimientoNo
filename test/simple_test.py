import cv2
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Cargar el modelo y el escalador
model = load('../models/digit_recognizer')
scaler = load('../models/scaler.joblib')

# Inicializar la captura de video
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def prediction(image, model, scaler):
    img = cv2.resize(image, (28, 28))
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)
    predict = model.predict(img)
    return predict[0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_copy = frame.copy()

    bbox_size = (100, 100)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]

    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)  # Correct thresholding

    digit = prediction(thresh, model, scaler)
    
    # Añadir la predicción del dígito en el frame
    cv2.putText(frame_copy, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Dibujar el cuadro de la región de interés (ROI)
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 2)
    cv2.imshow("input", frame_copy)
    cv2.imshow("cropped", thresh)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
