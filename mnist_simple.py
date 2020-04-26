import tensorflow as tf
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.summary()


print(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}, '
      f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

print(f'x_train[:1].shape={x_train[:1].shape}')

predictions = model(x_train[:1]).numpy()
print(f'predictions={predictions}')

print(f'softmax of predictions={tf.nn.softmax(predictions).numpy()}')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

print('About to evaluate')
model.evaluate(x_test, y_test, verbose=2)
