# CORY

"**C**ar **Ori**entation Convolutional Network"

A convolutional neural network aimed at determining the compass direction of cars.

Future plans are to detect cars and to determine driving direction on roads.

## Datasets

### EPFL
Access dataset through: [EPFL](http://cvlab.epfl.ch/data/pose)
#### Details
Labels were not provided, and so they were calculated by looking at the picture taking rate versus the rotation rate.  
- Image sizes are (376,250)
- 20 image sequences
#### Notes:
- Labels are sometimes within 1-2° of true label since times were truncated to whole seconds.

## Networks

* Transfer Learning on AlexNet
* Histogram of Oriented Gradients Feedforward Network

Both networks learn car pose angles via a discretization and probability distribution approach according to Hara, Vemulapalli, and Chellappa (2017).
