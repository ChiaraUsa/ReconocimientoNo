# Importaciones necesarias
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from joblib import load
import warnings

# Ignorar advertencias específicas para una mejor legibilidad
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead. SymbolDatabase.GetPrototype() will be removed soon.')


# Cargar el modelo de reconocimiento de dígitos y el escalador
model = load('../models/digit_recognizer')
scaler = load('../models/scaler.joblib')

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
# Obtener el ancho y el alto del video
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
    img = cv2.resize(image, (28, 28))  # Redimensionar la imagen a 28x28 píxeles
    img = img.flatten().reshape(1, -1)  # Aplanar la imagen y redimensionarla para el modelo
    img = scaler.transform(img)  # Escalar la imagen
    predict = model.predict(img)  # Realizar la predicción
    return predict[0]  # Devolver la predicción

# Inicializar variables para el modo de pintura
points = [deque(maxlen=1024)]  # Lista de deques para almacenar puntos
index = 0  # Índice para los puntos
color = (0, 0, 0)  # Color de los puntos (negro)
paintWindow = np.zeros((471, 636, 3)) + 255  # Crear una ventana de pintura blanca
# Dibujar un rectángulo y añadir texto "CLEAR" en la ventana de pintura
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
mpHands = mp.solutions.hands  # Inicializar mediapipe para la detección de manos
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)  # Configurar para detectar una mano
mpDraw = mp.solutions.drawing_utils  # Utilidad para dibujar las conexiones de las manos

# Variable para alternar entre modos (cámara o pintura)
mode = 'camera'

def destroy_windows(windows):
    """
    Cierra las ventanas de OpenCV especificadas en la lista proporcionada.

    Parámetros:
    -----------
    windows : lista de str
        Lista de nombres de las ventanas que se desean cerrar.

    Función:
    --------
    Itera sobre cada nombre de ventana en la lista y cierra la ventana correspondiente utilizando
    cv2.destroyWindow(window). Si ocurre un error (por ejemplo, si la ventana no existe), 
    el error es capturado y se ignora para evitar la interrupción del programa.
    """
    for window in windows:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass

# Bucle Principal y Visualización
while True:
    ret, frame = cap.read()  # Capturar frame de video
    if not ret:
        break

    if mode == 'camera':
        frame_copy = frame.copy()  # Hacer una copia del frame
        bbox_size = (100, 100)  # Tamaño del cuadro delimitador
        bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
                (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]  # Calcular las coordenadas del cuadro delimitador
        img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]  # Recortar la imagen
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
        _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)  # Binarizar la imagen
        digit = prediction(thresh, model, scaler)  # Predecir el dígito
        cv2.putText(frame_copy, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Mostrar la predicción en la imagen
        cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 2)  # Dibujar el cuadro delimitador
        cv2.imshow("Input", frame_copy)  # Mostrar la imagen original con el cuadro delimitador
        cv2.imshow("Cropped", thresh)  # Mostrar la imagen recortada y binarizada
        destroy_windows(["Output", "Paint"])  # Destruir las ventanas de salida y pintura

    elif mode == 'paint':
        x, y, c = frame.shape  # Obtener las dimensiones del frame
        frame = cv2.flip(frame, 1)  # Voltear la imagen horizontalmente
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir la imagen a RGB
        frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)  # Dibujar un rectángulo en la imagen
        cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)  # Añadir texto "CLEAR" en la imagen
        result = hands.process(framergb)  # Procesar la imagen para detectar manos

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)  # Escalar la coordenada x
                    lmy = int(lm.y * 480)  # Escalar la coordenada y
                    landmarks.append([lmx, lmy])  # Añadir las coordenadas a la lista de landmarks
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)  # Dibujar las conexiones de la mano
            fore_finger = (landmarks[8][0], landmarks[8][1])  # Coordenadas del dedo índice
            center = fore_finger  # Centro de la mano
            thumb = (landmarks[4][0], landmarks[4][1])  # Coordenadas del pulgar
            cv2.circle(frame, center, 3, (0, 255, 0), -1)  # Dibujar un círculo en el dedo índice

            if (thumb[1] - center[1] < 30):  # Si el pulgar está cerca del índice, crear un nuevo deque
                points.append(deque(maxlen=512))
                index += 1
            elif center[1] <= 65:  # Si el índice está en la parte superior de la pantalla, limpiar la ventana de pintura
                if 40 <= center[0] <= 140:
                    points = [deque(maxlen=512)]
                    index = 0
                    paintWindow[67:, :, :] = 255
            else:
                points[index].appendleft(center)  # Añadir el centro al deque de puntos
        else:
            points.append(deque(maxlen=512))  # Si no hay mano detectada, crear un nuevo deque
            index += 1

        for j in range(len(points)):
            for k in range(1, len(points[j])):
                if points[j][k - 1] is None or points[j][k] is None:
                    continue
                cv2.line(frame, points[j][k - 1], points[j][k], color, 5)  # Dibujar línea en el frame
                cv2.line(paintWindow, points[j][k - 1], points[j][k], color, 10)  # Dibujar línea en la ventana de pintura

        bbox_size = (200, 200)  # Tamaño del cuadro delimitador
        bbox = [(int(636 // 2 - bbox_size[0] // 2), int(471 // 2 - bbox_size[1] // 2)),
                (int(636 // 2 + bbox_size[0] // 2), int(471 // 2 + bbox_size[1] // 2))]  # Calcular las coordenadas del cuadro delimitador
        cv2.rectangle(frame, bbox[0], bbox[1], (0, 255, 0), 2)  # Dibujar el cuadro delimitador

        backup = paintWindow.copy()  # Hacer una copia de la ventana de pintura

        if index > 0 and len(points[index - 1]) > 1:  # Si hay puntos dibujados
            img_cropped = paintWindow[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]  # Recortar la imagen
            img_cropped = np.array(img_cropped, dtype=np.uint8)  # Convertir la imagen a uint8
            img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)  # Convertir la imagen a escala de grises
            _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)  # Binarizar la imagen
            digit = prediction(thresh, model, scaler)  # Predecir el dígito
            cv2.putText(paintWindow, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Mostrar la predicción en la ventana de pintura
            cv2.imshow("Paint", paintWindow)  # Mostrar la ventana de pintura
            cv2.waitKey(500)  # Esperar 500 ms
            paintWindow = backup  # Restaurar la ventana de pintura

        cv2.imshow("Output", frame)  # Mostrar el frame
        cv2.imshow("Paint", paintWindow)  # Mostrar la ventana de pintura
        destroy_windows(["Input", "Cropped"])  # Destruir las ventanas de entrada y recortada

    key = cv2.waitKey(1)  # Esperar una tecla durante 1 ms
    if key == ord('q'):
        break  # Salir del bucle si se presiona 'q'
    elif key == ord('c'):
        mode = 'paint' if mode == 'camera' else 'camera'  # Alternar entre modos si se presiona 'c'

# Liberar recursos
cap.release()  # Liberar la captura de video
cv2.destroyAllWindows()  # Cerrar todas las ventanas
