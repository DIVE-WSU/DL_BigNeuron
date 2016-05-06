#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sample_selector.hpp"
namespace caffe {

template <typename Dtype>
SampleSelector<Dtype>::SampleSelector(const LayerParameter& param)
:param_(param)
{
   InitRand();
   ProcessLabelSelectParam();
   for(int i=0;i<label_prob_weight_blob_.count();++i){
     float l_prob =label_prob_weight_blob_.cpu_data()[i];
     LOG(INFO)<<"lable class "<<"[" <<i<<"]  weight =" << l_prob;
   }
}


template <typename Dtype>
void SampleSelector<Dtype>::ProcessLabelSelectParam(){
  if(!param_.has_label_select_param()){
      balancing_label_ =false;
      vector<int> shape(1,1000);
      label_prob_weight_blob_.Reshape(shape);
      caffe_set(label_prob_weight_blob_.count(), static_cast<float>(1), label_prob_weight_blob_.mutable_cpu_data());
      return;
    }

    //LOG(INFO)<<"num_label ="<<num_labels_;

  num_labels_      =param_.label_select_param().num_labels() ;
  balancing_label_ =param_.label_select_param().balance();
  map2order_label_ =param_.label_select_param().reorder_label();
  bool has_prob_file=param_.label_select_param().has_class_prob_mapping_file();

  vector<int> shape(1,num_labels_);
  label_prob_weight_blob_.Reshape(shape);

  if(balancing_label_)
     {
    CHECK_EQ(balancing_label_,has_prob_file)<<"sample  class blanceing requires a class probability file provided ... ";
    const string&  label_prob_map_file = param_.label_select_param().class_prob_mapping_file();
    ReadLabelProbMappingFile(label_prob_map_file);
    //LOG(INFO)<<"Done ReadLabelProbMappingFile";
    ComputeLabelSkipRate();
    //LOG(INFO)<<"compute_label_skip_rate()";
    }
}

template <typename Dtype>
void SampleSelector<Dtype>::ReadLabelProbMappingFile(const string& source){
  LabelProbMappingParameter lb_param;
  LOG(INFO)<<"read prob from file : "<< source;
  ReadProtoFromTextFileOrDie(source, &lb_param);
  ignore_rest_of_label_ 				= 	lb_param.ignore_rest_of_label();
  rest_of_label_mapping_ 				= 	lb_param.rest_of_label_mapping();
  rest_of_label_mapping_label_	=	  lb_param.rest_of_label_mapping_label();
  rest_of_label_prob_					  =	  lb_param.rest_of_label_prob();
  num_labels_with_prob_  				= 	lb_param.label_prob_mapping_info_size();
  if(param_.label_select_param().has_num_top_label_balance())
    num_top_label_balance_ =param_.label_select_param().num_top_label_balance();
  else
    num_top_label_balance_ =  num_labels_with_prob_;
    if(ignore_rest_of_label_)
      caffe_set(label_prob_weight_blob_.count(), static_cast<float>(0), label_prob_weight_blob_.mutable_cpu_data());
    else
      caffe_set(label_prob_weight_blob_.count(), static_cast<float>(1), label_prob_weight_blob_.mutable_cpu_data());

  CHECK_GE(num_top_label_balance_,1);
  CHECK_GE(num_labels_with_prob_,num_top_label_balance_);
  CHECK_GE(num_labels_,2);
  CHECK_GE(num_labels_,num_labels_with_prob_);
  LOG(INFO)<<"rest_of_label_mapping_  = "<<rest_of_label_mapping_<<" "<<rest_of_label_mapping_label_;
  label_prob_map_.clear();
  label_mapping_map_.clear();
  
  for (int i=0;i<num_labels_with_prob_;++i){
        const LabelProbMapping&   label_prob_mapping_param = lb_param.label_prob_mapping_info(i);
        int   label 			=	label_prob_mapping_param.label();
        float lb_prob			=   label_prob_mapping_param.prob();
        int   mapped_label ;
        if(label_prob_mapping_param.has_map2label())
           mapped_label =   label_prob_mapping_param.map2label();
        else
          mapped_label = label ;
        label_prob_map_[label]	=	lb_prob;
        label_mapping_map_[label] =   mapped_label;
		LOG(INFO)<< "label map :"<<label<<"--->"<<label_mapping_map_[label];
      }
	  LOG(INFO)<< "label_prob_map_ size =" <<label_prob_map_.size();
 }
typedef std::pair<int, float> PAIR;
struct CmpByValue {
  bool operator()(const PAIR& lhs, const PAIR& rhs)
  {return lhs.second > rhs.second;}
 };

 template <typename Dtype>
 void SampleSelector<Dtype>::ComputeLabelSkipRate(){
      float scale_factor =0;
      vector<PAIR> label_prob_vec(label_prob_map_.begin(), label_prob_map_.end());
      sort(label_prob_vec.begin(), label_prob_vec.end(), CmpByValue()); //prob descend order;
      float bottom_prob=label_prob_vec[num_top_label_balance_-1].second;
      if(!ignore_rest_of_label_&&rest_of_label_prob_>0){
         scale_factor =bottom_prob < rest_of_label_prob_? 1.0/bottom_prob: 1.0/rest_of_label_prob_;
      }
      else
      {
        scale_factor =1.0/(bottom_prob==0?0.00001:bottom_prob) ;
      }
      LOG(INFO)<<" scale_factor =  "<< scale_factor;
      LOG(INFO)<<" bottom_prob =   " << bottom_prob;
      LOG(INFO)<<"label_prob_vec.size = "<<label_prob_vec.size();
      label_prob_map_.clear();
      // remove the class that has prob lower than top k classes;
      for(int i=0;i<num_top_label_balance_;++i)
      {
        int lb= label_prob_vec[i].first;
        float prob =label_prob_vec[i].second;
        label_prob_map_[lb]=prob;
        // mapping the label based on freq
        if(map2order_label_)
        {
         label_mapping_map_[lb]=i;
        }
      }

      // Init the rest of label class

      for (int i=0;i<num_labels_ ;++i)
      {
          int label =i;
          if(label_prob_map_.find(label)==label_prob_map_.end())
             {
                 if(ignore_rest_of_label_){
                   label_prob_map_[label]	=	0;
                   }
                 else
                    {
                    int rest_of_label =(num_labels_-num_top_label_balance_);
                    if(rest_of_label>0)
                     label_prob_map_[label]	=	rest_of_label_prob_;///rest_of_label;
                      //LOG(INFO)<<"rest_of_label_prob["<<label<<"]=" <<label_prob_map_[label];
                   }
                 if(rest_of_label_mapping_){
                   label_mapping_map_[label] =   rest_of_label_mapping_label_;
                   //LOG(INFO)<<"rest_of_label_mapping_["<<label<<"]=" <<label_mapping_map_[label];
                   }
                   else
                     label_mapping_map_[label] =   label;
                   // if reorder label is set, override the rest_of_label_mapping_
                 if(map2order_label_){
                     label_mapping_map_[label] =   num_top_label_balance_;
                 }
            }
       }
      //auto iterSkipRate  =label_mapping_map_.begin();
      vector<int> shape_p = label_prob_weight_blob_.shape();
      LOG(INFO)<<"size of prob = "<< shape_p.size();
      float* prob= label_prob_weight_blob_.mutable_cpu_data();

      std::map<int,float>::iterator iterProb;
       for (iterProb = label_prob_map_.begin(); iterProb !=label_prob_map_.end(); ++iterProb) {
         label_skip_rate_map_[iterProb->first] =ceil(iterProb->second*scale_factor);
         CHECK_LT(iterProb->first,  num_labels_);
         prob[iterProb->first]=1.0/label_skip_rate_map_[iterProb->first];
         //LOG(INFO)<<"label_skip_rate_map_["<<iterProb->first<<"]=" <<label_skip_rate_map_[iterProb->first];
        }


 }

 template <typename Dtype>
 bool SampleSelector<Dtype>::AcceptGivenLabel(const int label){
         //
       //balancing_label_
       if(!balancing_label_)
           return true;
       //label_skip_rate_map_[1]=1;
	   //label_skip_rate_map_[0]=5000;
       if (label_skip_rate_map_[label] ==0)
          return false;
       int reminder =PrefetchRand()%label_skip_rate_map_[label];
	   //if(reminder ==0)
	    //LOG(INFO)<<"label_skip_rate_map_["<<label<<"] =" <<label_skip_rate_map_[label]<<"reminder =="<<reminder;
       if(reminder ==0)
           return true;
       else
         return false;
 }
 template <typename Dtype>
 const float*  SampleSelector<Dtype>:: Get_Label_prob_gpu_data(){
   return label_prob_weight_blob_.gpu_data();
 }
  template <typename Dtype>
  const float*  SampleSelector<Dtype>:: Get_Label_prob_cpu_data(){
   return label_prob_weight_blob_.cpu_data();
  }
 
 template <typename Dtype>
 int SampleSelector<Dtype>::GetConvertedLabel(const int label){
       if(!balancing_label_)
       return label;
    else
      return label_mapping_map_[label];
 }
 
 template <typename Dtype>
 void SampleSelector<Dtype>:: Compute_label_prob_fromBlob(Blob<Dtype>* labelBlob){
	  caffe_set(label_prob_weight_blob_.count(), static_cast<float>(0), label_prob_weight_blob_.mutable_cpu_data());
	  float* prob= label_prob_weight_blob_.mutable_cpu_data();
	  size_t count =labelBlob->count();
	  int count_labels[1000]={0};
	  int max_label=0;
	  int min_count  =9999999;
	  const Dtype* lable_p =labelBlob->cpu_data();
	  for(size_t i=0;i<count;++i){
		  int label =static_cast<int>(lable_p[i]);
		  label=label_mapping_map_[label];
		  //if(label==2)
		  //LOG(INFO)<<"there is label 2";
		  count_labels[label]++;
		  max_label =max_label<label?label:max_label;
	  }
	  for (int i =0;i<=max_label;i++){
		  if (min_count>count_labels[i]){min_count=count_labels[i];}
	  }
	  CHECK_GE(min_count,1)<<"smallest number of label among classes must great then 0";
	  
	  for (int i =0;i<=max_label;i++)
	  {
		  CHECK_GE(count_labels[i],1);
		  prob[i]=static_cast<float>(min_count)/static_cast<float>(count_labels[i]);
	 }
    // return label_prob_weight_blob_.gpu_data();
 }
 

 template <typename Dtype>
 unsigned int SampleSelector<Dtype>::PrefetchRand(){
   CHECK(rng_);
   caffe::rng_t* rng =
       static_cast<caffe::rng_t*>(rng_->generator());
   return (*rng)();
 }
 template <typename Dtype>
 void SampleSelector<Dtype>::InitRand(){
   //if (needs_rand) {
	   rng_.reset();
     const unsigned int rng_seed = caffe_rng_rand();
        rng_.reset(new Caffe::RNG(rng_seed));
   //} else {
    //    rng.reset();
   //}
 }

 INSTANTIATE_CLASS(SampleSelector);
 }
