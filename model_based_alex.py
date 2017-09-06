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
epochs = 1
batch_size = 128

# load the EPFL Car Rotation dataset
data = epfl_data.Data()

train_samples = data.samples[:1873, :]
random.shuffle(train_samples)

validation_samples = data.samples[1873:,:]


images = []
targets = []
for sample in train_samples:

    # change color coding as RGB is expected by the remainder of the code
    name = sample[0]
    image = cv.imread(name)
    # change color coding as RGB is expected by the remainder of the code
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # scale the pose angle into [-0.5 0.5]
    target = float(sample[1])/360.0 - 0.5
    images.append(image)
    targets.append(target)

X_train = np.asarray(images)
y_train = np.asarray(targets)

images = []
targets = []
for sample in validation_samples:

    # change color coding as RGB is expected by the remainder of the code
    name = sample[0]
    image = cv.imread(name)
    # change color coding as RGB is expected by the remainder of the code
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # scale the pose angle into [-0.5 0.5]
    target = float(sample[1])/360.0 - 0.5
    images.append(image)
    targets.append(target)

X_val = np.asarray(images)
y_val = np.asarray(targets)


features = tf.placeholder(tf.float32, (None, 250, 376, 3))
labels = tf.placeholder(tf.float32, None)
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer 

fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
print(shape)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

mse = tf.losses.mean_squared_error(labels, logits)
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

        loss = sess.run(loss_op, feed_dict={features: X_batch, labels: y_batch})
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
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss= eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("")
