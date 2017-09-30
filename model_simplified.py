import cv2 as cv
import numpy as np
import epfl_data2 as epfl_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import losses
from keras import backend as K


data = epfl_data.Data(1,6)

height = 250
width = 376

training_images = np.zeros((1327,250,376,1))
test_images = np.zeros((972,250,376,1))
training_tgts = []
test_tgts = []

for d in range(len(data.samples[0])):
    image = cv.imread(data.samples[0][d])
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = np.expand_dims(image,2)
    if d <= 1326:
        training_images[d] = image
        training_tgts.append(data.samples[1][d])
    else:
        if d-1326 == 972:
            break
        else:
            test_images[d-1326] = image
            test_tgts.append(data.samples[1][d])

########LOSSES##########
def MAE(y_true, y_pred):
    # from Ghodrati 2014
    return -K.minimum(K.abs(y_pred-y_true), (2*np.pi)-(K.abs(y_pred-y_true)))

def circular_loss(y, y_pred):
    loss = 180 - K.abs(K.abs(y_pred - y) - 180)
    return loss

def cos_distance(y_true, y_pred):
    # from github
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)


for i in range(len(training_tgts)):
    training_tgts[i] = np.deg2rad(training_tgts[i])

for i in range(len(test_tgts)):
    test_tgts[i] = np.deg2rad(test_tgts[i])


model = Sequential()
model.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(height, width, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


model.compile(optimizer="adam", loss=losses.cosine)

model.fit(training_images, np.asarray(training_tgts), batch_size=10, epochs=5, verbose=1)

# model.save('model.h5')

model.summary()

test = np.expand_dims(test_images[2],axis=0)
print(test.shape)
predictions = []
predictions.append(model.predict(np.expand_dims(test_images[2],axis=0), batch_size=1, verbose=1))
print(test_tgts[2])
print(predictions[0])
print(test_tgts[20])
predictions.append(model.predict(np.expand_dims(test_images[20],axis=0), batch_size=1, verbose=1))
print(predictions[1])
print(test_tgts[200])
predictions.append(model.predict(np.expand_dims(test_images[200],axis=0), batch_size=1, verbose=1))
print(predictions[2])
# print(MAE(test_tgts[2],predictions[0]), MAE(test_tgts[20],predictions[1]), MAE(test_tgts[200],predictions[2]))