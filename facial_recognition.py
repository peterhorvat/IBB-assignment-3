import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import numpy as np
from termcolor import colored
from sklearn.metrics import roc_curve
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt

train_dir = "db/training"
validation_dir = "db/validation"
IMAGE_SIZE = 224
BATCH_SIZE = 64
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

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

#print(train_generator.class_indices)
#print(val_generator.class_indices)

# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')
# base_model.trainable = False
# model = tf.keras.Sequential([
#     base_model,  # 1
#     tf.keras.layers.Conv2D(16, 3, activation='relu'),
#     tf.keras.layers.Dropout(0.2),  # 3
#     tf.keras.layers.GlobalAveragePooling2D(),  # 4
#     tf.keras.layers.Dense(512,activation='relu'),
#     tf.keras.layers.Dense(6, activation='softmax')  # 5
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(2,2),activation='relu',input_shape=IMG_SHAPE),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(16,(2,2),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32,(2,2),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(6,activation='softmax')])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_generator,
                    epochs=18,
                    #steps_per_epoch=10,
                    callbacks=[callbacks],
                    validation_data=val_generator)
# base_model.trainable = True
#
# fine_tune_at = 100
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable = False
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(1e-5),
#               metrics=['accuracy'])
# history_fine = model.fit(train_generator,
#                          epochs=10,
#                          callbacks=[callbacks],
#                          #steps_per_epoch=10,
#                          validation_data=val_generator)
dir_path = 'db/testing/'

tf = s = ad = aj = ah = ahp = ba = bm = 0
for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+i, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    tf += 1
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    predict_class = (np.argmax(model.predict(images), axis=1))
    #print(np.argmax(predict_class, axis=1))
    if predict_class[0] == 0:
        if i.startswith('ad'):
            print(colored("Alexandra Daddario", 'green'))
            s += 1
            ad += 1
        else:
            print(colored("Alexandra Daddario", 'red'))
    elif predict_class[0] == 1:
        if i.startswith('aj'):
            print(colored("Angelina Jolie", 'green'))
            s += 1
            aj += 1
        else:
            print(colored("Angelina Jolie", 'red'))
    elif predict_class[0] == 2:
        if i.startswith('ah'):
            print(colored("Anne Hathaway", 'green'))
            s += 1
            ah += 1
        else:
            print(colored("Anne Hathaway", 'red'))
    elif predict_class[0] == 3:
        if i.startswith('ahp'):
            print(colored("Anthony Hopkins", 'green'))
            s += 1
            ahp += 1
        else:
            print(colored("Anthony Hopkins", 'red'))
    elif predict_class[0] == 4:
        if i.startswith('ba'):
            print(colored("Ben Affleck", 'green'))
            s += 1
            ba += 1
        else:
            print(colored("Ben Affleck", 'red'))
    elif predict_class[0] == 5:
        if i.startswith('bm'):
            print(colored("Bill Murray", 'green'))
            s += 1
            bm += 1
        else:
            print(colored("Bill Murray", 'red'))
print("############## RESULTS ##############\n"
      "# Alexandra D. : {0:.2f}%             #\n"
      "# Angelina J.  : {1:.2f}%             #\n"
      "# Anne H.      : {2:.2f}%             #\n"
      "# Anthony H.   : {3:.2f}%             #\n"
      "# Ben A.       : {4:.2f}%             #\n"
      "# Bill M.      : {5:.2f}%             #\n"
      "# Accuracy     : {6:.2f}% ({7}/{8})     #\n"
      "#####################################"
      "".format((ad/10)*100, (aj/10)*100, (ah/10)*100, (ahp/10)*100, (ba/10)*100, (bm/10)*100, (s/tf)*100, s, tf))
