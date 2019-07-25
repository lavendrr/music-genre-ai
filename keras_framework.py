#loading the dataset
from keras.dataset import #name of dataset here

(train_data, train_labels), (test_data, test_labels) =
'''datasetname'''.load_data(num_secs = 30)


#vectorize data here with Librosa


#define the model
import numpy as np

from keras import models
from keras import layers


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(30,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense('''numberofpossibleoutputs''', activation='softmax'))


#compiling the model
model.compile(optimizer = 'rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])


#setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]


#training the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 20
                    batch_size = 512
                    validation_data = (x_val, y_val))


#plotting the training and validation loss
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#plotting the training and validation accuracy
plt.clf() #clears the figure

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
