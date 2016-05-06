#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  int count = 0;
  int total_label[1000]={0};
  int correct_lb[1000]={0};
  for (int i = 0; i < outer_num_; ++i) {

    for (int j = 0; j < inner_num_; ++j) {
      //const int label_value =
      //    static_cast<int>(bottom_label[i * inner_num_ + j]);
	  int label_value =
          static_cast<int>(bottom_label[i * inner_num_ + j]);
	  if (label_value ==2) label_value =0;
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {

          ++accuracy;
          CHECK_LT(label_value,num_labels);
		  correct_lb[label_value]++;
          /* if(label_value==1)
            {correct_lb_1++;}
          else if(label_value==0){
            correct_lb_0++;
          }else{
          LOG(INFO)<<"unkonwn label = "<<label_value;
          } */
          //LOG(INFO)<<"acc prob  " <<"["<<k<<"] = "<<bottom_data_vector[k].first;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
		//LOG(INFO)<< "predicT_lb== " << bottom_data_vector[k].second <<" lb =" <<label_value;
      }
      ++count;
	  total_label[label_value]++;

    }

  }
  
 /*  for(int i=0;i<num_labels;++i){
	   LOG(INFO)<<"label "<<i<< " = "  <<correct_lb[i]<<"/"<<total_label[i];
  } */
  //  LOG(INFO)<<"label 4  C/T  :  "<<correct_lb[4]<<"/"<<total_label[4]<<"  label 3  C/T  :  "<<correct_lb[3]<<"/"<<total_label[3]<<"  label 2  C/T  :  "<<correct_lb[2]<<"/"<<total_label[2]<<"  label 1  C/T  :  "<<correct_lb[1]<<"/"<<total_label[1]<<"  label 0 C/T : "<<correct_lb[0]<<"/"<< total_label[0] <<  "  count = "<<count;

  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
