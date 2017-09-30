import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

from alexnet import AlexNet
import epfl_data 
from vonmiseskde import VonMisesKDE

tf.logging.set_verbosity(tf.logging.INFO)

def flip_angle(angle):
    if angle == 180:
        return 0
    elif angle == 0:
        return 180
    else:
        return 360-angle

nb_rotations = 2
nb_classes = 8
epochs = 5
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
softmaxes = []
mse = []
for i in range(nb_rotations):
    logit = tf.nn.xw_plus_b(maxpool5, orientation_W[i], orientation_b[i])
    logits.append(logit)
    softmaxes.append(tf.nn.softmax(logit))
    mse.append(tf.losses.sparse_softmax_cross_entropy(labels=labels[i], logits=logit))

loss_op = tf.reduce_mean(np.sum(mse))

predictions = []
matches = []
accuracies = []
for i in range(nb_rotations):
    predictions.append(tf.argmax(tf.nn.softmax(logits[i]),1))
    matches.append(tf.equal(predictions[i], labels[i]))
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
    all_pred = []
    all_lab = []
    all_softmax = []
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
        softm, pred, lab, acc, loss = sess.run([softmaxes, predictions, labels, accuracies, loss_op], feed_dict=feed_dict)
        all_pred.append(pred)
        all_lab.append(lab)

        all_softmax.append(softm)

        total_loss += (loss * X_batch.shape[0])
        for i in range(nb_rotations):
            total_acc[i] += (acc[i] * X_batch.shape[0])

    return all_softmax, all_pred, all_lab, total_loss/X.shape[0], total_acc/X.shape[0]

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

        softm, pred, lab, val_loss, acc = eval_on_data(X_val, y_val, sess)
        print("Epoch {}".format(i+1))
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss = {}".format(val_loss))
        print("Accuracies = {}".format(acc))
        
        flat_softm = [item for sublist in softm for item in sublist] 
        flat_softm = np.concatenate(flat_softm)#.ravel()
        flat_softm = flat_softm.reshape((-1,nb_rotations*nb_classes))

        ### prints to examine classification results ###
        #flat_lab = [item for sublist in lab for item in sublist] 
        #flat_lab = np.concatenate(flat_lab).ravel()
        #print(flat_lab)
        #flat_pred = [item for sublist in pred for item in sublist] 
        #flat_pred = np.concatenate(flat_pred).ravel()
        #print(np.transpose(np.vstack((flat_lab,flat_pred))))
        #num_hits = np.sum(np.equal(flat_pred,flat_lab))
        #print("{} von {} richtig".format(num_hits,len(flat_pred)))
        #print(np.asarray(lab).flatten(), np.asarray(pred).flatten())

discrete_orientations = data.binnies[1873:]

results = np.zeros((len(discrete_orientations),2))

# Kernel density estimator
kappa = 10
for i in range(len(flat_softm)):
    kde = VonMisesKDE(np.deg2rad(discrete_orientations[i]), weights=flat_softm[i], kappa=kappa)

    # Input for test points
    test_x = np.linspace(0, 2*math.pi, 720)

    # Display posterior estimate
    plt.plot(test_x, kde.evaluate(test_x), zorder=20)
    plt.title('von Mises density for a sample image')
    maxi = np.argmax(kde.evaluate(test_x))
    #print("Radians: {}, Degree: {}, True Deg: {}".format(test_x[maxi], np.rad2deg(test_x[maxi]), data.samples[1873+i][1]))
    results[i,0] = np.rad2deg(test_x[maxi])
    results[i,1] = data.samples[1873+i][1]
    plt.xlabel("Orientation")
    plt.xlim(0, 2*math.pi)
    plt.ylim(0, 1)
    plt.show()

def circular_loss(y, y_pred):
    loss = 180 - np.absolute(np.absolute(y_pred - y) - 180)
    return loss

print(results)

error = circular_loss(results[:,1], results[:,0])

plt.hist(error, 30)
plt.title("Distribution of Error")
plt.xlabel("Error Magnitude")
plt.ylabel("Frequency")
plt.show()

print("Mean error: {}".format(np.mean(error)))
print("Median error: {}".format(np.median(error)))
