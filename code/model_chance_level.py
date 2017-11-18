import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import losses
from keras import backend as K
import math

from sklearn import utils
import epfl_data 

# load the EPFL Car Rotation dataset
data = epfl_data.Data(1,6)

train_samples = data.samples[:1873]
random.shuffle(train_samples)

validation_samples = data.samples[1873:]

def circ_mean_squared_error(y_true,y_pred):
    deg_true = (y_true + 0.5) * 360
    deg_pred = (y_pred + 0.5) * 360
    loss = (180 - K.abs(K.abs(deg_pred - deg_true) - 180)) / 360
    return K.mean(K.cast(loss, K.floatx()))

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            targets = []
            for batch_sample in batch_samples:

                # change color coding as RGB is expected by the remainder of the code
                name = batch_sample[0]
                image = cv.imread(name)
                # change color coding as RGB is expected by the remainder of the code
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                # scale the pose angle into [-0.5 0.5]

                target = float(batch_sample[1])/360.0 - 0.5
                images.append(image)
                targets.append(target)

            X_train = np.asarray(images)
            y_train = np.asarray(targets)
            yield utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=10)
validation_generator = generator(validation_samples, batch_size=10)

ch, row, col = 3, 250, 376  # image format

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras import backend as Back

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
model.add(Lambda(lambda x: x/255.0 - 0.5,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
model.add(Convolution2D(16, (8, 8), strides=(4, 4), activation='relu', input_shape=(row, col, ch)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))


model.compile(optimizer="adam", loss=circ_mean_squared_error)
# Train the model using the generator to feed training data
model.fit_generator(train_generator, steps_per_epoch=
            len(train_samples), validation_data=validation_generator,
            validation_steps=len(validation_samples), epochs=3)

evalu = model.evaluate_generator(validation_generator, 10)

print(model.metrics_names)
print('Avg error on validation data : {}'.format(evalu))

# doesnt work for some reason , dont know why
#pred = model.predict_generator(validation_generator, 10)
#print(pred)

model.save('model.h5')

model.summary()