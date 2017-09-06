import argparse
import cv2 as cv
import numpy as np
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import epfl_data 
import matplotlib.pyplot as plt
import math

model = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Pose')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )

    args = parser.parse_args()

    import keras.backend as K
    def angle_difference(x, y):
        """
        Calculate minimum difference between two angles.
        """
        return 180 - abs(abs(x - y) - 180)

    def angle_error(y_true, y_pred):
        """
        Calculate the mean diference between the true angles
        and the predicted angles. Each angle is represented
        as a binary vector.
        """
        diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
        return K.mean(K.cast(K.abs(diff), K.floatx()))

    import keras.losses

    keras.losses.angle_error = angle_error

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)


    # load the EPFL Car Rotation dataset
    data = epfl_data.Data(1,6)
    validation_samples = data.samples[1873:]

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
        #target = np.sin(np.radians(float(sample[1]) - 180.0))
        #target = float(sample[1])
        images.append(image)
        targets.append(target)

    X_validation = np.asarray(images)
    y_validation = np.asarray(targets)


    #### This is just a very simple hard coded example and should be extended ###

    #ame = './test_images/testa.jpg'
    # this is looked up in epfl_targets.csv
    #true_pose = 142.5
    #image = cv.imread(name)
    # change color coding as RGB is expected by the remainder of the code
    #image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    pred_pose = model.predict(X_validation, batch_size=10)
    #print(pred_pose.squeeze().shape)

    # recover to degree space
    #pred_pose = (pred_pose + 0.5) * 360
    #print('Predicted angle: {} and true angle: {}'.format(pred_pose, true_pose))
    #pred_deg = (pred_pose.squeeze() + 0.5) * 360
    #true_deg = (y_validation + 0.5) * 360
    pred_deg = (pred_pose.squeeze()+0.5)*360
    true_deg = (y_validation+0.5)*360
    output = zip(pred_deg, true_deg) 
    plt.hist(pred_deg,30)
    plt.show()
    #avg_err = np.sum(180 - abs(abs(pred_deg - true_deg) - 180)) / len(true_deg)
    print('Predicted angle: {} '.format(np.asarray(output)))
