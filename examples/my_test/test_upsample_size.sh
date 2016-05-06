#!/bin/bash
GLOG_logtostderr=1 /tempspace/tzeng/caffe_nd_sense_segmetation/.build_release/tools/caffe.bin train --solver=test_upsample_solver.prototxt --gpu 7