import tensorflow as tf
import numpy as np
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

# load model
filepath = "quick_planet_model"
model = tf.keras.models.load_model(filepath, compile = True)

print(model.summary())

# Test
test_image_path = "planets/Moon/Moon (122).jpg"
image = tf.keras.utils.load_img(test_image_path,target_size=(144,256))
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = model.predict(input_arr)

labels = np.array(train_ds.class_names)
result = np.array(predictions[0])

print(labels)
print(result)

planets_list = ['Earth', 'Jupiter', 'MakeMake', 'Mars', 'Mercury', 'Moon', 'Neptune', 'Pluto', 'Saturn', 'Uranus', 'Venus']

print("\n__________________\n","max: ",np.argmax(result))
print(f'{fg(196)}{bg(55)}{ planets_list[np.argmax(result)], predictions[0].max()*100 }{attr(0)}')



plt.scatter(planets_list, predictions[0])
plt.show()

plt.imshow(tf.keras.utils.load_img(test_image_path,target_size=(144,256), color_mode="grayscale"))
plt.show()