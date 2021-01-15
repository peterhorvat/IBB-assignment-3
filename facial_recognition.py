import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

train_dir = "db/training"
validation_dir = "db/validation"
IMAGE_SIZE = 224
BATCH_SIZE = 3
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, BATCH_SIZE)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy') > .98) & (logs.get('val_accuracy') > .9):
            print("Reached 98% accuracy: [Cancelling training]!")
            self.model.stop_training = True
callbacks = myCallback()

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)
train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)
val_generator = data_generator.flow_from_directory(
    validation_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)

for image_batch, label_batch in train_generator:
    break

print(train_generator.class_indices)
print(val_generator.class_indices)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,  # 1
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # 3
    tf.keras.layers.GlobalAveragePooling2D(),  # 4
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 5
])

model.compile(optimizer=tf.keras.optimizers.Adam(),  # 1
              loss='categorical_crossentropy',  # 2
              metrics=['accuracy'])  # 3
history = model.fit(train_generator,
                    steps_per_epoch=5,
                    epochs=30,
                    callbacks=[callbacks],
                    validation_data=val_generator,)

base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-5),
              metrics=['accuracy'])
history_fine = model.fit(train_generator,
                         epochs=5,
                         callbacks=[callbacks],
                         steps_per_epoch=5,
                         validation_data=val_generator

                         )
dir_path = 'db/test/'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    #plt.imshow(img)
    #plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    predict_class = (np.argmax(model.predict(images), axis=1))
    print(i)
    #print(np.argmax(predict_class, axis=1))
    if predict_class[0] == 0:
        print("Andrew Garfield")
    elif predict_class[0] == 1:
        print("Goldie Hawn")
    elif predict_class[0] == 2:
        print("Kurt Russell")
    else:
        print("Wyatt Russell")
