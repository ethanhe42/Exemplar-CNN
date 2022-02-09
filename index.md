Code used in the paper "Discriminative Unsupervised Feature Learning with Convolutional Neural Networks", NIPS 2014

Copyright (c) 2014 Alexey Dosovitskiy

------------------------------
Terms of use
------------------------------

The code is provided for research purposes only. Any commercial
use is prohibited. If you are interested in a commercial use, please 
contact the copyright holder. 

If you used this program in your research work, you should cite the 
following publication:

Alexey Dosovitskiy, Jost Tobias Springenberg, Martin Riedmiller and Thomas Brox,
Discriminative Unsupervised Feature Learning with Convolutional Neural Networks,
Advances in Neural Information Processing Systems 27 (NIPS 2014).

The code is distributed WITHOUT ANY WARRANTY.

------------------------------
Dependencies
------------------------------

For network training and testing we used Caffe software ( http://arxiv.org/abs/1408.5093, http://caffe.berkeleyvision.org/ ), this is a must-have dependency.

For testing you'll also need liblinear ( http://www.csie.ntu.edu.tw/~cjlin/liblinear/ )

For Spatial Pyramid Pooling (when testing on Caltech-101) we used pooling code from Matthew Zeiler's Adaptive Deconvolutional Networks ( http://www.matthewzeiler.com/ ). This can also in principle be done using caffe or some matlab code.

------------------------------
Description
------------------------------

We provide code for creating the training data, training and testing of the networks; as well as network configuration files and pre-trained networks.

For training a network, see code/training/demo_train_net.sh ( after correcting some paths in the demo and in train_nn_pretrain.sh ). The script allows pretraining with a diferent number of classes, which we do not actually need any more.

For testing, see code/testing/demo_test_net.m (matlab), you also need to correct some paths and download some datasets.

For dataset creation, see code/make_data/augment_patches_distr.m .

For converting data from matlab to leveldb or lmdb, see code/training/demo_convert_data.sh ( after correcting some paths in convert_data.sh ). 

The configuration files include 'template' files which should be used with train_nn_pretrain.sh script, and test.prototxt files which may be used for testing with a pretrained network. Some of the files are in 'old' format which was previously used by caffe.

------------------------------
Bugs
------------------------------

Please report any bugs to Alexey Dosovitskiy ( dosovits@cs.uni-freiburg.de )
