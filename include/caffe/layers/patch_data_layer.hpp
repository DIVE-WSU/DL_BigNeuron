#ifndef CAFFE_PATCH_DATA_LAYER_HPP_
#define CAFFE_PATCH_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/patch_sampler.hpp"

namespace caffe {

template <typename Dtype>
class PatchDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit PatchDataLayer(const LayerParameter& param);
  virtual ~PatchDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // DataLayer uses patch_sampler instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "PatchData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void load_batch(Batch<Dtype>* batch);
   PatchSampler<Dtype> patch_sampler_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
