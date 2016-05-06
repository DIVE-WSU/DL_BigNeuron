#!/bin/bash
GLOG_logtostderr=1 /tempspace/tzeng/caffe_nd_sense_segmetation/.build_release/tools/caffe.bin train --solver=bn_ResNet_solver_Single_Lb.prototxt
