import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import cv2 as cv
import numpy as np

from alexnet import AlexNet
import epfl_data 

nb_classes = 6
epochs = 10
batch_size = 128

# load the EPFL Car Rotation dataset
data = epfl_data.Data(1,6)

train_samples = data.samples[:1873]
random.shuffle(train_samples)

validation_samples = data.samples[1873:]


images = []
labels = []
for sample in train_samples:

    # change color coding as RGB is expected by the remainder of the code
    name = sample[0]
    image = cv.imread(name)
    # change color coding as RGB is expected by the remainder of the code
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    label = sample[2][0]    
    images.append(image)
    
    labels.append(label)

X_train = np.asarray(images)
y_train = np.asarray(labels)

images = []
labels = []
for sample in validation_samples:

    # change color coding as RGB is expected by the remainder of the code
    name = sample[0]
    image = cv.imread(name)
    # change color coding as RGB is expected by the remainder of the code
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    label = sample[2][0]    
    images.append(image) 
    labels.append(label)


X_val = np.asarray(images)
y_val = np.asarray(labels)


features = tf.placeholder(tf.float32, (None, 250, 376, 3))
labels = tf.placeholder(tf.int64, [None])
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer 

fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)

fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
print(labels.shape, logits.shape)
mse = tf.losses.sparse_softmax_cross_entropy(labels, logits)
loss_op = tf.reduce_mean(mse)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
init_op = tf.initialize_all_variables()

def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss = sess.run(loss_op, feed_dict={features: X_batch, labels: np.argmax(y_batch, axis=1)})
        total_loss += (loss * X_batch.shape[0])

    return total_loss/X.shape[0]

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            #print(np.argmax(y_train[offset:end], axis=1))
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: np.argmax(y_train[offset:end], axis=1)})

        val_loss= eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("")
