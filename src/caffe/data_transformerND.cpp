#include <string>
#include <vector>

#include "caffe/data_transformerND.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformerND<Dtype>::DataTransformerND(const TransformationNDParameter& param)
    : param_(param){
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}
template<typename Dtype>
Dtype DataTransformerND<Dtype>::ReadOnePoint(Blob<Dtype>* input_blob, vector<int>& pointCoord){
  int input_shape_dims=input_blob->num_axes();
  const vector<int>& input_shape =input_blob->shape();
  CHECK_GE(input_shape_dims,3);
  const int input_num = input_shape[0];
  const int input_channels = input_shape[1];
  CHECK_EQ(input_channels,1);
  CHECK_EQ(input_num,1);
  if(pointCoord.size()!=(input_shape_dims-2) && pointCoord.size()!=input_shape_dims)
  LOG(FATAL)<<"dim of input point coords must equal to dim of input blob, or 2 dim smaller if num and channel info is ignored";

  int pt_index=pointCoord[0];
  for (int n=1;n<pointCoord.size();++n){
      //  if(n==0){
      //    pt_index = pointCoord[n];
      //  }else{
       pt_index*=input_shape[2+n];
       pt_index+=pointCoord[n];
     //}
  }
  int input_count=input_blob->count();
  CHECK_GE(pt_index,0);
  CHECK_LE(pt_index,input_count);
  const Dtype* input_data =input_blob->cpu_data();
  Dtype pt_value=input_data[pt_index];
  //LOG(INFO)<<"center reding idex =" <<pt_index;
  return pt_value;
}
template<typename Dtype>
const CropCenterInfo<Dtype>  DataTransformerND<Dtype>::PeekCropCenterPoint(Blob<Dtype>* input_blob){
  //PeekCropCenterPoint assum that input is a label array, thus it has only one channel.

  //for(int i=0;input_blob->shape().size();++i)
    //  LOG(INFO)<<"input blob shape size =" <<input_blob->shape().size();
  bool padding =param_.padding();
  bool crop =param_.has_crop_shape();
  int input_shape_dims=input_blob->num_axes();
  CropCenterInfo<Dtype> crop_center_info;//(new CropCenterInfo<Dtype>);
  //CropCenterInfo<Dtype> crop_center_info;//(new CropCenterInfo<Dtype>);
  CHECK_GE(input_shape_dims,3);
  const vector<int>& input_shape =input_blob->shape();
  vector<int> tranform_shape;
  const int input_num = input_shape[0];
  const int input_channels = input_shape[1];
  CHECK_EQ(input_channels,1);
  CHECK_EQ(input_num,1);
  int crop_shape_dims=param_.crop_shape().dim_size();
  CHECK_EQ(crop_shape_dims,input_shape_dims-2);
  // assume that the number
  tranform_shape.push_back(1);
  tranform_shape.push_back(1);
  CHECK_EQ(crop_shape_dims,input_shape_dims-2);
  for(int i=0;i<crop_shape_dims;i++){
      tranform_shape.push_back(param_.crop_shape().dim(i));
  }
  vector<int> nd_off(crop_shape_dims,0);

     for(int i=0;i<nd_off.size();i++){
       if(crop)
          if(padding){
              CHECK_GE(input_shape[i+2],0);
              nd_off[i] = Rand(input_shape[i+2])-tranform_shape[i]/2;
            }
          else
          {
              nd_off[i] = Rand(input_shape[i+2] - tranform_shape[i] + 1);
              CHECK_GE(input_shape[i+2],0);
            }
        else
          nd_off[i] = (input_shape[i+2]  - tranform_shape[i]) / 2;

          //for(int i=0;input_blob->shape().size();++i)
        //LOG(INFO)<<"patch offset  =" <<nd_off[i] ;
      //  w_off = (input_width - crop_size) / 2;
    }



  int center_index=1;
  for (int n=0;n<nd_off.size();++n){
       if(n==0){
         center_index = nd_off[n];
          // LOG(INFO)<<"n=0 center inx "<<center_index;
       }else{
       center_index*=input_shape[2+n];
       center_index+=nd_off[n];
     }
  }
  int input_count=input_blob->count();
  CHECK_GE(center_index,0);
  CHECK_LE(center_index,input_count);
  const Dtype* input_data =input_blob->cpu_data();
  Dtype center_value=input_data[center_index];
  crop_center_info.nd_off=nd_off;
  crop_center_info.value =center_value;

  //LOG(INFO)<<"nd_off num_aix ="<<crop_center_info.nd_off.size();
  return crop_center_info;
  //TransformationNDParameter_PadMethod_ZERO;

}

template<typename Dtype>
void DataTransformerND<Dtype>::Transform(Blob<Dtype>* input_blob,
                                                Blob<Dtype>* transformed_blob){
//  CropCenterInfo<Dtype> c_info= PeekCropCenterPoint(input_blob);

  CropCenterInfo<Dtype> c_info= PeekCropCenterPoint(input_blob);
  Transform(input_blob, transformed_blob,c_info.nd_off);

}
template<typename Dtype>
void DataTransformerND<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob, const vector<int>& off_set) {
  int input_shape_dims=input_blob->num_axes();
//  bool padding =param_.padding();
  int offset_axis = off_set.size();
  CHECK_GE(input_shape_dims,3);
  CHECK_EQ(offset_axis,input_shape_dims-2);
  bool crop =param_.has_crop_shape();
  const vector<int>& input_shape =input_blob->shape();
  vector<int> transform_shape;
  const int input_num = input_shape[0];
  const int input_channels = input_shape[1];
  transform_shape.push_back(input_num);
  transform_shape.push_back(input_channels);
  int crop_shape_dims=param_.crop_shape().dim_size();
  CHECK_EQ(crop_shape_dims,input_shape_dims-2);
  for(int i=0;i<crop_shape_dims;i++){
      transform_shape.push_back(param_.crop_shape().dim(i));
  }

  if (transformed_blob->count() == 0) {
    // Initialize transformed_blob with the right shape.
    if (crop) {
      transformed_blob->Reshape(transform_shape);
    } else {
      transformed_blob->Reshape(input_shape);
    }
  }
 //return ;
  vector<int> new_transform_shape = transformed_blob->shape();
  //const int num = new_transform_shape[0];
  const int channels = new_transform_shape[1];
  // const int height = transformed_blob->height();
  // const int width = transformed_blob->width();
  const size_t trans_data_size = transformed_blob->count();

  //CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  //CHECK_GE(input_height, height);
  //CHECK_GE(input_width, width);


  const Dtype scale = param_.scale();
  // do mirro for each of dimention respectively
  //const bool do_mirror = param_.mirror() && Rand(crop_shape_dims+1);
//  const bool has_mean_values = mean_values_.size() > 0;

  //int h_off = 0;
  //int w_off = 0;
  // vector<int> nd_off(crop_shape_dims,0);
  //
  // if(crop){
  //   nd_off=off_set;
  // }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  //int start_spatial_aixs =2;
 for(size_t p=0;p<trans_data_size;++p){
   // revise compute the dat index in the input blob;
   vector<int> nd_point;
   vector<int>::iterator it;
   size_t pre_aixs_len =0;
   int data_axis_idx =0;
   for(int i=transform_shape.size()-1;i>0;--i){

     if(i==transform_shape.size()-1){
        data_axis_idx=p%transform_shape[i]+off_set[i-2];
        //if(do_mirror)
        //    transform_shape[i]-(data_axis_idx+1);
        //nd_point.push_back(data_axis_idx);
        pre_aixs_len=transform_shape[i];
     }else{
       data_axis_idx= i-2>=0 ?
                               (p/pre_aixs_len)%transform_shape[i] +off_set[i-2]
                               :(p/pre_aixs_len)%transform_shape[i];
        pre_aixs_len*=transform_shape[i];
     }
       it =nd_point.begin();
       nd_point.insert(it, data_axis_idx);
       //nd_point.push_back(data_axis_idx);
   }
  data_axis_idx=p/pre_aixs_len;
  it =nd_point.begin();
  nd_point.insert(it, data_axis_idx);

 size_t data_idx=0;

   for (int n=0;n<nd_point.size();++n){
        if(n==0){
          data_idx = nd_point[n];
        }else{
        data_idx*=input_shape[n];
        data_idx+=nd_point[n];
      }
   }

   size_t input_count=input_blob->count();
   const Dtype* input_data =input_blob->cpu_data();
   bool data_in_pad_space =(data_idx>=0 && data_idx<input_count);
   if(data_in_pad_space)
     transformed_data[p]=input_data[data_idx];
   else
     transformed_data[p]=0;
 }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal( trans_data_size, scale, transformed_data);
  }

}

template <typename Dtype>
void DataTransformerND<Dtype>::Transform(Blob<Dtype>* input_blob,
                                        Blob<Dtype>* transformed_blob,
                                        const vector<int>& off_set,
                                        const vector<int>& crop_shape)
{

    int input_shape_dims=input_blob->num_axes();
    int offset_axis = off_set.size();
    int crop_shape_axis =crop_shape.size();
    CHECK_EQ(offset_axis,crop_shape_axis);
    CHECK_GE(input_shape_dims,3);
    CHECK_EQ(offset_axis,input_shape_dims-2);
    const vector<int>& input_shape =input_blob->shape();
    vector<int> transform_shape;
    const int input_num = input_shape[0];
    const int input_channels = input_shape[1];
    CHECK_EQ(input_num ,1)<<"num of input must be 1 ";
    //CHECK_EQ(input_channels,1)<<"num of channels must be 1";
    //const int channels = new_transform_shape[1];
    transform_shape =crop_shape;
    transform_shape.insert(transform_shape.begin(),input_channels);
    transform_shape.insert(transform_shape.begin(),input_num);


  //  CHECK_EQ(crop_shape_axis,input_shape_dims-2);
    //LOG(INFO)<<"transform start...";
    if (transformed_blob->count() == 0) {
        transformed_blob->Reshape(transform_shape);
    }
    const Dtype scale = param_.scale();
      // do mirro for each of dimention respectively
     //const bool do_mirror = param_.mirror() && Rand(crop_shape_axis+1);
   //  const bool has_mean_values = mean_values_.size() > 0;
    // vector<int> nd_off(crop_shape_axis,0);
    const Dtype* input_data =input_blob->cpu_data();
    Dtype* transformed_data = transformed_blob->mutable_cpu_data();
    const size_t trans_data_size = transformed_blob->count();

   vector<int> nd_point(input_shape_dims,0);
    for(size_t p=0;p<trans_data_size;++p){
     // revise compute the dat index in the input blob
        int index =p;
     for(int i=transform_shape.size()-1;i>=off_set.size()-1;--i){
         nd_point[i]=index%transform_shape[i]+off_set[i-2];
         //if(do_mirror)
         //     //    nd_point[i]=transform_shape[i]-nd_point[i];
         index/=transform_shape[i];
        //  if(i==transform_shape.size()-1){
        //     data_axis_idx=p%transform_shape[i]+off_set[i-2];
        //     //LOG(INFO)<<"  data_axis_idx ="<<data_axis_idx<<"  off_set[i-2]="<<off_set[i-2];
        //     //if(do_mirror)
        //     //    transform_shape[i]-(data_axis_idx+1);
        //     //nd_point.push_back(data_axis_idx);
        //     pre_aixs_len=transform_shape[i];
        //  }else{
        //
        //       data_axis_idx= i-2>=0 ?
        //       (p/pre_aixs_len)%transform_shape[i] +off_set[i-2]
        //       :(p/pre_aixs_len)%transform_shape[i];
        //       pre_aixs_len*=transform_shape[i];
        //  }
        // //  it =nd_point.begin();
        // //  nd_point.insert(it, data_axis_idx);
        // nd_point[i]=data_axis_idx;
         //LOG(INFO)<<"nd_point = "<< data_axis_idx;
     }
      for(int i=off_set.size()-2;i>=0;--i){
        nd_point[i]=index%transform_shape[i];
        index/=transform_shape[i];
      }

    //LOG(INFO)<<"computed nd_points...";
    //  data_axis_idx=(p/pre_aixs_len);
    //  it =nd_point.begin();
    //  nd_point.insert(it, data_axis_idx);

    bool data_in_pad_space =false;
      size_t  data_idx = nd_point[0];
     for (int n=1;n<nd_point.size();++n){
         if(nd_point[n]<0 || nd_point[n]>input_shape[n]-1){
           data_in_pad_space =true;
		   //LOG(INFO)<<"Pad out range set to 0"<<nd_point[n];
           break;
         }
        data_idx*=input_shape[n];
        data_idx+=nd_point[n];
     }
      transformed_data[p]=data_in_pad_space?0:input_data[data_idx];
    }
    //   if(!data_in_pad_space)
    //      transformed_data[p]=input_data[data_idx];
    //    else
    //      transformed_data[p]=0;
    //  }
    if (scale != Dtype(1)) {
      LOG(INFO) << "Scale: " << scale;
      caffe_scal( trans_data_size, scale, transformed_data);
    }
  //  LOG(INFO)<<"transform done..";

}
template <typename Dtype>
void DataTransformerND<Dtype>::ApplyMean(Blob<Dtype>* input_blob,
                  Blob<Dtype>* transformed_blob){

  vector<int> src_shape  = input_blob->shape();
  int iput_axes          = input_blob->num_axes();

  int channels_axis =1;
  int num_ch_means = mean_values_.size();
  int num = src_shape[0];
  int channels =src_shape[channels_axis];
  size_t data_size=input_blob->count(channels_axis+1);
  const Dtype* input_data =input_blob->cpu_data();
  Dtype* trans_data =transformed_blob->mutable_cpu_data();
  if(input_blob!=transformed_blob)
    transformed_blob->Reshape(src_shape);
  else
    caffe_copy(transformed_blob->count(),input_data,trans_data);

  if(param_.mean_value_size()==0) return;

  CHECK_GE(iput_axes,3);

//  CHECK_EQ(channels,num_ch_means);
// do not compute if mean_values == 0
  if ((num_ch_means >= 1)&&(mean_values_[0]!=0)){
  int count = 0;
  for(int i=0;i<num;++i)
    for(int c=0;c<channels;++c){
      for(int d=0;d<data_size;++d)
          trans_data[count++]-=mean_values_[c];
    }
  }

}


template <typename Dtype>
void DataTransformerND<Dtype>::InitRand() {
  const bool needs_rand = param_.has_crop_shape();
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformerND<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformerND);

}  // namespace caffe
