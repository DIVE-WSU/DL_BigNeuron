#!/bin/bash
GLOG_logtostderr=1 /tempspace/tzeng/caffe_nd_sense_segmetation/.build_release/tools/caffe.bin train --solver=test_solver.prototxt
