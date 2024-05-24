# ReconocimientoNo

Este proyecto implementa un sistema de reconocimiento de dígitos utilizando una cámara en tiempo real. El sistema utiliza un modelo SVM para la clasificación de dígitos y está diseñado para segmentar y reconocer dígitos de imágenes capturadas por la cámara.

## Autores
- Chiara Valenzuela.
- Mitchell Bermin.
- Oscar Miranda.

## Estructura del Proyecto

```plaintext
ReconocimientoNo/
├── test_recognition.py
├── .gitignore
├── algorithm.py
├── camara.py
├── model.py
├── recognize_n.py
├── requirements.txt
└── README.md
```

## Instalación

### Prerrequisitos

- Python 3.7+
- cv2 (OpenCV)
- numpy
- joblib
- tensorflow
- scikit-learn

### Clonar el repositorio

```bash
git clone https://github.com/ChiaraUsa/ReconocimientoNo.git
cd ReconocimientoNo
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Ejecutar la aplicación de reconocimiento

```bash
py recognize_n.py
```

Esto abrirá una ventana de video en tiempo real desde la cámara conectada. La región de interés (ROI) para el reconocimiento de dígitos está marcada con un rectángulo azul en el centro de la pantalla. Los dígitos reconocidos se mostrarán en la parte superior de la ROI.

### Pruebas Unitarias

Para ejecutar las pruebas unitarias, usa el siguiente comando:

```bash
py -m unittest test_recognition.py
```

## Estructura del Código

- `recognize_n.py`: Contiene la lógica principal para capturar video, procesar imágenes, segmentar dígitos y reconocerlos.
- `algorithm.py`: Contiene el código relacionado con el algoritmo de reconocimiento.
- `model.py`: Contiene el código para el entrenamiento del modelo SVM.
- `camara.py`: Código adicional para la captura y procesamiento de imágenes.
- `test_recognition.py`: Contiene las pruebas unitarias para el código de reconocimiento.

## Modelos

- `svm_digit_classifier.joblib`: El modelo SVM entrenado para la clasificación de dígitos.
- `scaler.joblib`: El escalador estándar utilizado para normalizar los datos de entrada antes de la clasificación.

## Contribuciones

Las contribuciones son bienvenidas. Siéntete libre de abrir un issue o enviar un pull request.
