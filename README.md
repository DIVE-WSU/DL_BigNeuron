# DL_BigNeuron
This code is modifided from master version of caffe(https://github.com/BVLC/caffe), particularlly for 3D neuron image segmentaion in BigNeuron Project (https://http://alleninstitute.org/bigneuron/about/)

Functionalities:
- 1. Fully Convolutional  networks  with 3D data segmentataion.
- 2. Selecting random patchs from input Volum  into batch to train the networks, eliminating the need to create patch files for training.
- 3. Enabling the sampling  balanced calsses for batch training by set probability of each class in th file
- 4. setting weighted loss for each class
- 5. fast image based predictions.
    }
