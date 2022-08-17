import os
import cv2
import keras
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from datetime import datetime
from matplotlib import pyplot
from keras.applications import InceptionResNetV2
start_time = datetime.now()
pre_model = InceptionResNetV2(weights="imagenet",include_top=False,input_shape=(150,150,3))
#pre_model.summary()

train_cats = []
train_dogs = []
test_cats = []
test_dogs = []

for filename in os.listdir('./cats_dogs_dataset/dataset/training_set/cats'):
    image = cv2.imread(os.path.join('./cats_dogs_dataset/dataset/training_set/cats', filename))
    image = cv2.resize(image, (150,150))
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    if image is not None:
        train_cats.append(np.asarray(image))
print('1')
for filename in os.listdir('./cats_dogs_dataset/dataset/training_set/dogs'):
    image = cv2.imread(os.path.join('./cats_dogs_dataset/dataset/training_set/dogs', filename))
    image = cv2.resize(image, (150,150))
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    if image is not None:
        train_dogs.append(np.asarray(image))
print('2')
for filename in os.listdir('./cats_dogs_dataset/dataset/test_set/cats'):
    image = cv2.imread(os.path.join('./cats_dogs_dataset/dataset/test_set/cats', filename))
    image = cv2.resize(image, (150, 150))
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    if image is not None:
        test_cats.append(np.asarray(image))
print('3')
for filename in os.listdir('./cats_dogs_dataset/dataset/test_set/dogs'):
    image = cv2.imread(os.path.join('./cats_dogs_dataset/dataset/test_set/dogs', filename))
    image = cv2.resize(image, (150, 150))
    image = tf.keras.applications.mobilenet.preprocess_input(image)
    if image is not None:
        test_dogs.append(np.asarray(image))
print('4')
# label encoding: cat = 1, dog = 0

# Create n x 1 label arrays for each set of images, n = # of data points for that label
labels_cats_train = np.full((len(train_cats)), 1)
labels_dogs_train = np.full((len(train_dogs)), 0)
labels_cats_test = np.full((len(test_cats)), 1)
labels_dogs_test = np.full((len(test_dogs)), 0)
print('5')
# Concatenate labels into one array
labels_train = np.hstack((labels_cats_train, labels_dogs_train))
labels_test = np.hstack((labels_cats_test, labels_dogs_test))
print('6')
# Concatenate the data arrays into one array in the same order as labels
dataset_train = np.vstack((train_cats, train_dogs))
dataset_test = np.vstack((test_cats, test_dogs))
print('7')
print("Preparing CNN")

# Build the CNN model
model = models.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1,activation = 'sigmoid'))
model.summary()
pre_model.trainable=False

# Compiling and Training the model
print("Training Model")
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])


history = model.fit(dataset_train, labels_train, epochs=5,
                        validation_data=(dataset_train, labels_train))

yp_test = model.predict(dataset_test)
yp_test = np.argmax(yp_test, axis=1)
cm_test = confusion_matrix(labels_test, yp_test)
t3 = ConfusionMatrixDisplay(cm_test)
t3.plot()
plt.show()
plt.clf()


test_loss, test_acc = model.evaluate(dataset_test, labels_test, verbose=2)
print(test_acc)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
