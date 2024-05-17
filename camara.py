import cv2

# Inicializar la captura de video desde la cámara (0 es usualmente la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    # Si no se pudo obtener el frame, salir del loop
    if not ret:
        print("No se pudo recibir el frame (stream end?). Saliendo...")
        break

    # Mostrar el frame en una ventana llamada 'frame'
    cv2.imshow('frame', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
