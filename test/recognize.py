# Importación de Módulos y Configuración de Advertencias
import cv2
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

# Bucle Principal y Visualización
while True:
    ret, frame = cap.read() # Lee un frame de video
    if not ret: #Si no se puede leer el frame, rompe el bucle.
        break

    frame_copy = frame.copy() # Crea una copia del frame para la visualización.

    # Definición de la caja delimitadora (ROI)
    bbox_size = (100, 100) # Tamaño de la caja delimitadora.

    # Coordenadas de la caja delimitadora centrada en el frame.
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]
    
    # Procesamiento de la imagen en la ROI
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] # Recorta la región de interés (ROI) del frame.
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY) # Convierte la imagen recortada a escala de grises.
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)  # Aplica umbralización binaria inversa a la imagen en escala de grises.

    # Predicción del dígito en la imagen umbralizada utilizando la función prediction.
    digit = prediction(thresh, model, scaler)
    
    # Añadir la predicción del dígito en el frame
    cv2.putText(frame_copy, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Dibujar el cuadro de la región de interés (ROI)
    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 2) # Dibuja la caja delimitadora en el frame.

    # Muestra el frame con la predicción y la imagen umbralizada en ventanas separadas.
    cv2.imshow("input", frame_copy) 
    cv2.imshow("cropped", thresh)

    # Control del bucle
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
