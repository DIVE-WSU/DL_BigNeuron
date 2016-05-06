# DL_BigNeuron
This code is modifided from master version of caffe(https://github.com/BVLC/caffe), particularlly for 3D neuron image segmentaion in BigNeuron Project (https://http://alleninstitute.org/bigneuron/about/)

Functionalities:
- 1. Fully Convolutional  networks  with 3D data segmentataion.
- 2  Fast image based predictions.
- 3. Selecting random patchs from 3d Volum  to build batch for networks training, eliminating the need to create patch files for training.
- 4. Enabling the sampling  balanced calsses for batch training by setting probability of each class in prototxt file
- 5. Setting weighted loss for each class
- 6. Including avaible inception-residual-fully convolutional archetecture, made particularlly for 3D neuron image segmentaion
