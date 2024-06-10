import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from joblib import load
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Cargar el modelo y el escalador
model = load('../models/digit_recognizer')
scaler = load('../models/scaler.joblib')

# Inicializar la captura de video
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def prediction(image, model, scaler):
    """
    Realiza la predicción del dígito en la imagen proporcionada utilizando el modelo y el escalador.

    Parámetros:
    -----------
    image : array-like
        La imagen de entrada que contiene el dígito.

    model : sklearn model
        El modelo de clasificación entrenado.

    scaler : sklearn scaler
        El escalador entrenado para estandarizar las características.

    Retorna:
    --------
    predict[0] : int
        La predicción del dígito en la imagen.
    """
    img = cv2.resize(image, (28, 28))
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)
    predict = model.predict(img)
    return predict[0]

# Inicializar variables para el modo de pintura
points = [deque(maxlen=1024)]
index = 0
color = (0, 0, 0)
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Variable para alternar entre modos
mode = 'camera'

def destroy_windows(windows):
    for window in windows:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass

# Bucle Principal y Visualización
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if mode == 'camera':
        frame_copy = frame.copy()
        bbox_size = (100, 100)
        bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
                (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]
        img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
        digit = prediction(thresh, model, scaler)
        cv2.putText(frame_copy, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow("Input", frame_copy)
        cv2.imshow("Cropped", thresh)
        destroy_windows(["Output", "Paint"])

    elif mode == 'paint':
        x, y, c = frame.shape
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        result = hands.process(framergb)

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            fore_finger = (landmarks[8][0], landmarks[8][1])
            center = fore_finger
            thumb = (landmarks[4][0], landmarks[4][1])
            cv2.circle(frame, center, 3, (0, 255, 0), -1)

            if (thumb[1] - center[1] < 30):
                points.append(deque(maxlen=512))
                index += 1
            elif center[1] <= 65:
                if 40 <= center[0] <= 140:
                    points = [deque(maxlen=512)]
                    index = 0
                    paintWindow[67:, :, :] = 255
            else:
                points[index].appendleft(center)
        else:
            points.append(deque(maxlen=512))
            index += 1

        for j in range(len(points)):
            for k in range(1, len(points[j])):
                if points[j][k - 1] is None or points[j][k] is None:
                    continue
                cv2.line(frame, points[j][k - 1], points[j][k], color, 2)
                cv2.line(paintWindow, points[j][k - 1], points[j][k], color, 2)

        bbox_size = (200, 200)
        bbox = [(int(636 // 2 - bbox_size[0] // 2), int(471 // 2 - bbox_size[1] // 2)),
                (int(636 // 2 + bbox_size[0] // 2), int(471 // 2 + bbox_size[1] // 2))]
        cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)

        backup = paintWindow.copy()

        if index > 0 and len(points[index - 1]) > 1:
            img_cropped = paintWindow[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
            img_cropped = np.array(img_cropped, dtype=np.uint8)
            img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
            digit = prediction(thresh, model, scaler)
            cv2.putText(paintWindow, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Paint", paintWindow)
            cv2.waitKey(500)
            paintWindow = backup

        cv2.imshow("Output", frame)
        cv2.imshow("Paint", paintWindow)
        destroy_windows(["Input", "Cropped"])

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        mode = 'paint' if mode == 'camera' else 'camera'

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
