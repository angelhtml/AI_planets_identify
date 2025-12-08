import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from colored import fg, bg, attr


val_ds = tf.keras.utils.image_dataset_from_directory(
  "./planets",
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(144, 256),
  batch_size=32)

train_ds = tf.keras.utils.image_dataset_from_directory(
  "./planets",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(144, 256),
  batch_size=32)



num_classes = 11

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=12
)

model.save("quick_planet_model")


"""

# load model
filepath = "quick_planet_model"
model = tf.keras.models.load_model(filepath, compile = True)

print(model.summary())

# Test
image = tf.keras.utils.load_img("planets/Pluto/Pluto (101).jpg",target_size=(144,256))
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)

labels = np.array(train_ds.class_names)
result = np.array(predictions[0])

print(labels)
print(result)

planets_list = ['Earth', 'Jupiter', 'MakeMake', 'Mars', 'Mercury', 'Moon', 'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus']

print("\n__________________\n","max: ",np.argmax(result))
print(f'{fg(196)}{bg(55)}{ planets_list[np.argmax(result)] }{attr(0)}')

"""