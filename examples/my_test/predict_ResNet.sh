#!/bin/bash
GLOG_logtostderr=1 /tempspace/tzeng/caffe_nd_sense_segmetation/.build_release/tools/predict_seg.bin deploy_bigneuron_ResNET_v2_small.prototxt \
patch_test_fcn_ResNET_5pool_snapshot_v2_iter_2413.caffemodel \
mean_file.binary \
/tempspace/tzeng/bigneuron/big_neuron_hackthon_tesla/data/hd5_valid.h5 \
/tempspace/tzeng/bigneuron/big_neuron_hackthon_tesla/data/predict_valid.h5 \
CPU \
5