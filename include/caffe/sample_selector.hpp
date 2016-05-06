#ifndef CAFFE_SAMPLE_SELECTOR_HPP_
#define CAFFE_SAMPLE_SELECTOR_HPP_

#include <map>
#include <string>
#include <vector>
#include <boost/thread.hpp>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
  template <typename Dtype>
  class SampleSelector {
   public:
    explicit SampleSelector(const LayerParameter& param);
    void ProcessLabelSelectParam();
    void ReadLabelProbMappingFile(const string& file_name);
    void ComputeLabelSkipRate();
    bool AcceptGivenLabel(const int label);
    int  GetConvertedLabel(const int label);
    const float*  Get_Label_prob_gpu_data();
	const float*  Get_Label_prob_cpu_data();
	void  Compute_label_prob_fromBlob(Blob<Dtype>* labelBlob);
  protected:
    void InitRand();
    unsigned int PrefetchRand();
    const LayerParameter param_;
    //const LabelSelectParameter param_;
    std::map <int, float> label_prob_map_;
    std::map <int, unsigned int> label_skip_rate_map_;
    std::map <int, int> label_mapping_map_;
    unsigned  int num_labels_;
    unsigned  int num_labels_with_prob_ ;
    bool      balancing_label_;
    unsigned int num_top_label_balance_;
    bool  rest_of_label_mapping_;
    bool  map2order_label_;
    bool  ignore_rest_of_label_;
    int   rest_of_label_mapping_label_;
    float rest_of_label_prob_;
    shared_ptr<Caffe::RNG> rng_;
    Blob<int> label_skip_rate_blob_;
    Blob<float> label_prob_weight_blob_;
  };
}
#endif
