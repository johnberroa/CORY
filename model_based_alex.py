import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import cv2 as cv
import numpy as np

from alexnet import AlexNet
import epfl_data 

tf.logging.set_verbosity(tf.logging.INFO)

def flip_angle(angle):
    if angle == 180:
        return 0
    elif angle == 0:
        return 180
    else:
        return 360-angle

nb_rotations = 3
nb_classes = 6
epochs = 200
batch_size = 32

# load the EPFL Car Rotation dataset
data = epfl_data.Data(nb_rotations, nb_classes)

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

    angle = sample[1]  
    label = sample[2]  
    images.append(image)
    labels.append(label)

    #print(label)
    # image augmentation
    #image_flipped = np.fliplr(image)
    #print("Angle: {}".format(angle))
    #angle_flipped = flip_angle(angle)
    #print(angle_flipped)
    #augmented_label = data._discretized_labels([angle])[0]
                
    #images.append(image_flipped)
    #labels.append(augmented_label)


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

    label = sample[2]    
    images.append(image) 
    labels.append(label)


X_val = np.asarray(images)
y_val = np.asarray(labels)


features = tf.placeholder(tf.float32, (None, 250, 376, 3))
labels = []
for i in range(nb_rotations):
    labels.append(tf.placeholder(tf.int64, [None]))
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer 

maxpool5 = AlexNet(resized, feature_extract=True)
maxpool5 = tf.stop_gradient(maxpool5)

# average pooling size 3 stride 1
#pooled = tf.nn.pool(fc7, (3,3), "AVG", 1)

shape = (maxpool5.get_shape().as_list()[-1], nb_classes)

orientation_W = []
orientation_b = []
for i in range(nb_rotations):
    orientation_W.append(tf.Variable(tf.truncated_normal(shape, stddev=0.0001)))
    orientation_b.append(tf.Variable(tf.zeros(nb_classes)))

logits = []
mse = []
for i in range(nb_rotations):
    logit = tf.nn.xw_plus_b(maxpool5, orientation_W[i], orientation_b[i])
    logits.append(logit)
    mse.append(tf.losses.sparse_softmax_cross_entropy(labels=labels[i], logits=logit))

loss_op = tf.reduce_mean(np.sum(mse))

matches = []
accuracies = []
for i in range(nb_rotations):
    matches.append(tf.equal(tf.argmax(tf.nn.softmax(logits[i]),1), labels[i]))
    accuracies.append(tf.reduce_mean(tf.cast(matches[i], tf.float32)))

# set learning rate to decrease by a factor of 10 after 2000 iterations
global_step = tf.Variable(0, trainable=False)
boundaries = [2000]
values = [0.000001, 0.0000001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
#var_list=[orientation_W, orientation_b]
train_op = opt.minimize(loss_op)
init_op = tf.global_variables_initializer()

def eval_on_data(X, y, sess):
    total_acc = np.zeros((nb_rotations))

    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        las = []
        for i in range(nb_rotations):
            las.append(np.argmax(y_batch[:,i,:], axis=1))


        feed_dict={i: d for i, d in zip(labels, las)}
        feed_dict[features] = X_batch
        acc, loss = sess.run([accuracies, loss_op], feed_dict=feed_dict)
        total_loss += (loss * X_batch.shape[0])
        total_acc[i] += (acc[i] * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

with tf.Session() as sess:
    sess.run(init_op)
    step_count = 0
    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            step_count += 1
            if(step_count % 100 == 0):
                print("Steps: {}".format(step_count))
            end = offset + batch_size

            las = []
            for i in range(nb_rotations):
                las.append(np.argmax(y_train[offset:end][:,i,:], axis=1))

            feed_dict={i: d for i, d in zip(labels, las)}
            feed_dict[features] = X_train[offset:end]        
            sess.run(train_op, feed_dict=feed_dict)

        val_loss, acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss = {}".format(val_loss))
        print("Accuracies = {}".format(acc))
