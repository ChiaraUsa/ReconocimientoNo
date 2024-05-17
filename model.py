import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar el conjunto de datos MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los datos
x_train, x_test = x_train / 255.0, x_test / 255.0

# Construir el modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo
model.evaluate(x_test, y_test, verbose=2)

# Guardar el modelo
model.save('mnist_model.h5')
