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

nb_classes = 6
epochs = 200
batch_size = 32

# load the EPFL Car Rotation dataset
data = epfl_data.Data(2,6)

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

    # image augmentation
    #image_flipped = np.fliplr(image)
    #print("Angle: {}".format(angle))
    #angle_flipped = flip_angle(angle)
    #print(angle_flipped)
    #augmented_label = data._discretized_labels([angle])[0][0]
                
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
labels0 = tf.placeholder(tf.int64, [None])
labels1 = tf.placeholder(tf.int64, [None])
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer 

maxpool5 = AlexNet(resized, feature_extract=True)
maxpool5 = tf.stop_gradient(maxpool5)

# average pooling size 3 stride 1
#pooled = tf.nn.pool(fc7, (3,3), "AVG", 1)

shape = (maxpool5.get_shape().as_list()[-1], nb_classes)

orientation_W = tf.Variable(tf.truncated_normal(shape, stddev=0.0001))
orientation_b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(maxpool5, orientation_W, orientation_b)
mse1 = tf.losses.sparse_softmax_cross_entropy(labels=labels0, logits=logits)

orientation_W2 = tf.Variable(tf.truncated_normal(shape, stddev=0.0001))
orientation_b2 = tf.Variable(tf.zeros(nb_classes))
logits2 = tf.nn.xw_plus_b(maxpool5, orientation_W2, orientation_b2)
mse2 = tf.losses.sparse_softmax_cross_entropy(labels=labels1, logits=logits2)


loss_op = tf.reduce_mean(mse1 + mse2)

match1 = tf.equal(tf.argmax(tf.nn.softmax(logits),1), labels0)
match2 = tf.equal(tf.argmax(tf.nn.softmax(logits2),1), labels1)


accuracy1 = tf.reduce_mean(tf.cast(match1, tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(match2, tf.float32))

# set learning rate to decrease by a factor of 10 after 2000 iterations
global_step = tf.Variable(0, trainable=False)
boundaries = [2000]
values = [0.000001, 0.0000001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = opt.minimize(loss_op, var_list=[orientation_W, orientation_b, orientation_W2, orientation_b2])
init_op = tf.global_variables_initializer()

def eval_on_data(X, y, sess):
    total_acc1 = 0
    total_acc2 = 0

    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        la0 = np.argmax(y_batch[:,0,:], axis=1)
        la1 = np.argmax(y_batch[:,1,:], axis=1)

        acc1, acc2, loss = sess.run([accuracy1, accuracy2, loss_op], feed_dict={features: X_batch, labels0: la0, labels1:la1})
        total_loss += (loss * X_batch.shape[0])
        total_acc1 += (acc1 * X_batch.shape[0])
        total_acc2 += (acc2 * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc1/X.shape[0], total_acc2/X.shape[0]

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

            la0 = np.argmax(y_train[offset:end][:,0,:], axis=1)
            la1 = np.argmax(y_train[offset:end][:,1,:], axis=1)
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels0: la0, labels1:la1})

        val_loss, acc1, acc2 = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss = {}".format(val_loss))
        print("Accuracy 1 = {}, accuracy 2 = {}".format(acc1, acc2))
