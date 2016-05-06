#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        PoolingParameter pooling_param =this->layer_param_.pooling_param();
        channel_axis_ = bottom[0]->CanonicalAxisIndex(pooling_param.channels_axis());
        first_spatial_axis_ = channel_axis_ + 1;
        const int num_axes = bottom[0]->num_axes();
        num_spatial_axes_ = num_axes - first_spatial_axis_;
        CHECK_GE(num_spatial_axes_, 0);
        if(num_spatial_axes_ ==2)
          LayerSetUp2D(bottom, top);
        else
          LayerSetUpND(bottom, top);

}

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUpND(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
      PoolingParameter pool_param = this->layer_param_.pooling_param();
      global_pooling_ = pool_param.global_pooling();
      vector<int> bottom_shape=bottom[0]->shape();
    //  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
      vector<int> spatial_dim_blob_shape(1,std::max(num_spatial_axes_, 1));
      // Setup filter kernel dimensions (kernel_shape_).
      kernel_shape_.Reshape(spatial_dim_blob_shape);
      stride_shape_.Reshape(spatial_dim_blob_shape);
      pad_shape_.Reshape(spatial_dim_blob_shape);
      bottom_d_shape_.Reshape(spatial_dim_blob_shape);
      int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
      int* stride_shape_data = stride_shape_.mutable_cpu_data();
      int* pad_shape_data    = pad_shape_.mutable_cpu_data();

      CHECK(!pool_param.has_kernel_size()|| !pool_param.has_kernel_h()||! pool_param.has_kernel_w())
      <<"With ND input data: kernel_size, kernel_h and kernel_w can't be set";
      CHECK(!pool_param.has_pad()|| !pool_param.has_pad_h()||!pool_param.has_pad_w())
      <<"With ND input data: pad, pad_h and pad_w can't be set";
      CHECK(!pool_param.has_stride()||!pool_param.has_stride_h()||!pool_param.has_stride_w())
      <<"With ND input data: stride, stride_h and stride_w can't be set";

      if (global_pooling_) {
        CHECK(!pool_param.kernel_shape_size())
          << "With Global_pooling: true Filter shape cannot specified";
          for(int i=0;i<num_spatial_axes_;++i){
            kernel_shape_data[i]=bottom_shape[first_spatial_axis_+i];
          }
        }else{
          const int num_kernel_dims = pool_param.kernel_shape_size();
          CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
              << "kernel_size must be specified once, or once per spatial dimension "
              << "(kernel_size specified " << num_kernel_dims << " times; "
              << num_spatial_axes_ << " spatial dims);";
            for (int i = 0; i < num_spatial_axes_; ++i) {
              kernel_shape_data[i] =
                  pool_param.kernel_shape((num_kernel_dims == 1) ? 0 : i);
            }
        }

      const int num_stride_dims = pool_param.stride_shape_size();
      CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
            num_stride_dims == num_spatial_axes_)
          << "stride must be specified once, or once per spatial dimension "
          << "(stride specified " << num_stride_dims << " times; "
          << num_spatial_axes_ << " spatial dims);";
      const int kDefaultStride = 1;
      for (int i = 0; i < num_spatial_axes_; ++i) {
        stride_shape_data[i] = (num_stride_dims == 0) ? kDefaultStride :
            pool_param.stride_shape((num_stride_dims == 1) ? 0 : i);
        CHECK_GT(stride_shape_data[i], 0) << "Stride dimensions must be nonzero.";
      }

      const int num_pad_dims = pool_param.pad_shape_size();
      CHECK(num_pad_dims==0|| num_pad_dims==1 || num_pad_dims==num_spatial_axes_)
            << "pad must be specified once, or once per spatial dimension "
            << "(pad specified " << num_pad_dims << " times; "
            << num_spatial_axes_ << " spatial dims);";
      const int kDefaultPad = 0;
      for (int i = 0; i < num_spatial_axes_; ++i) {
            pad_shape_data[i] = (num_pad_dims == 0) ? kDefaultPad :
            pool_param.pad_shape((num_pad_dims == 1) ? 0 : i);
            CHECK_LT(pad_shape_data[i], kernel_shape_data[i]);
          }
      if (global_pooling_) {
             for(int i=0;i<num_spatial_axes_;++i){
               CHECK(stride_shape_data[i] ==1 && pad_shape_data[i]==0)
                << "With Global_pooling: true; only pad = 0 and stride = 1";
             }
      }
      //vector<int> bottom_shape =botttom[0]->shape();
      int* bottom_d_shape_data =bottom_d_shape_.mutable_cpu_data();
      for (int i = 0; i < num_spatial_axes_; ++i){
           bottom_d_shape_data[i]=bottom_shape[first_spatial_axis_+i];}
  }

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp2D(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
        PoolingParameter pool_param = this->layer_param_.pooling_param();
        if (pool_param.global_pooling()) {
          CHECK(!(pool_param.has_kernel_size() ||
            pool_param.has_kernel_h() || pool_param.has_kernel_w()))
            << "With Global_pooling: true Filter size cannot specified";
        } else {
          CHECK(!pool_param.has_kernel_size() !=
            !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
            << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
          CHECK(pool_param.has_kernel_size() ||
            (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
            << "For non-square filters both kernel_h and kernel_w are required.";
        }
        CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
            && pool_param.has_pad_w())
            || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
            << "pad is pad OR pad_h and pad_w are required.";
        CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
            && pool_param.has_stride_w())
            || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
            << "Stride is stride OR stride_h and stride_w are required.";
        global_pooling_ = pool_param.global_pooling();
        if (global_pooling_) {
          kernel_h_ = bottom[0]->height();
          kernel_w_ = bottom[0]->width();
        } else {
          if (pool_param.has_kernel_size()==1) {
            kernel_h_ = kernel_w_ = pool_param.kernel_size();
          } else {
            kernel_h_ = pool_param.kernel_h();
            kernel_w_ = pool_param.kernel_w();
          }
        }
        CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
        CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
        if (!pool_param.has_pad_h()) {
          pad_h_ = pad_w_ = pool_param.pad();
        } else {
          pad_h_ = pool_param.pad_h();
          pad_w_ = pool_param.pad_w();
        }
        if (!pool_param.has_stride_h()) {
          stride_h_ = stride_w_ = pool_param.stride();
        } else {
          stride_h_ = pool_param.stride_h();
          stride_w_ = pool_param.stride_w();
        }
        if (global_pooling_) {
          CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
            << "With Global_pooling: true; only pad = 0 and stride = 1";
        }
        if (pad_h_ != 0 || pad_w_ != 0) {
          CHECK(this->layer_param_.pooling_param().pool()
              == PoolingParameter_PoolMethod_AVE
              || this->layer_param_.pooling_param().pool()
              == PoolingParameter_PoolMethod_MAX)
              << "Padding implemented only for average and max pooling.";
          CHECK_LT(pad_h_, kernel_h_);
          CHECK_LT(pad_w_, kernel_w_);
        }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        if(num_spatial_axes_ ==2)
          Reshape2D(bottom, top);
        else
          ReshapeND(bottom, top);

}

template <typename Dtype>
void PoolingLayer<Dtype>::ReshapeND(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
      vector<int> bottom_shape = bottom[0]->shape();
      channels_ = bottom_shape[channel_axis_];
      int num   = bottom_shape[0];
      int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
      const int* stride_shape_data = stride_shape_.cpu_data();
      const int* pad_shape_data = pad_shape_.cpu_data();
      vector<int> spatial_dim_blob_shape(1,std::max(num_spatial_axes_, 1));
      //LOG(INFO)<<"spatial_dim_blob_shape size  ="<<spatial_dim_blob_shape.size();

      //LOG(INFO)<<"kernell_shape_ axes "<< kernel_shape_.count();
      //int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
      if (global_pooling_) {
        kernel_shape_.Reshape(spatial_dim_blob_shape);
        kernel_shape_data = kernel_shape_.mutable_cpu_data();
        for(int i=0;i<num_spatial_axes_;++i){
          kernel_shape_data[i]=bottom_shape[first_spatial_axis_+i];
        }
      }

      pooled_d_shape_.Reshape(spatial_dim_blob_shape);
      int* pooled_d_shape_data = pooled_d_shape_.mutable_cpu_data();
      vector<int>  pooled_shape;
      for(int i=0;i<num_spatial_axes_;++i){
      pooled_d_shape_data[i] = static_cast<int>(ceil(static_cast<float>(
          bottom_shape[first_spatial_axis_+i]  + 2 * pad_shape_data[i] - kernel_shape_data[i]) / stride_shape_data[i])) + 1;

        //  static_cast<int>(ceil(static_cast<float>(
        //      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
      }

      bool any_axis_pad_not_zero =false;
      for(int i=0;i<num_spatial_axes_;++i){
        if(pad_shape_data[i]>0){
           any_axis_pad_not_zero =true;
           break;
         }
      }
      if (any_axis_pad_not_zero) {
        // If we have padding, ensure that the last pooling starts strictly
        // inside the image (instead of at the padding); otherwise clip the last.
        for(int i=0;i<num_spatial_axes_;++i){
          if ((  pooled_d_shape_data[i] - 1) * stride_shape_data[i] >= bottom_shape[first_spatial_axis_+i] + pad_shape_data[i]) {
            --pooled_d_shape_data[i];
          }
          CHECK_LT((pooled_d_shape_data[i] - 1) *stride_shape_data[i], bottom_shape[first_spatial_axis_+i] + pad_shape_data[i]);
        }
     }


      vector<int> out_put_shape ;//= pooled_shape_;
      pooled_data_length_ =1;
      for(int i=0;i<num_spatial_axes_;++i){
        pooled_data_length_*=pooled_d_shape_data[i];
        out_put_shape.push_back(pooled_d_shape_data[i]);
          //std::cout<<"poold_d_shape ="<<pooled_d_shape_data[i]<<std::endl;
      }
      out_put_shape.insert(out_put_shape.begin(),channels_);
      out_put_shape.insert(out_put_shape.begin(),num);

      top[0]->Reshape(out_put_shape);
      if (top.size() > 1) {
        top[1]->ReshapeLike(*top[0]);
      }

      // If max pooling, we will initialize the vector index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX && top.size() == 1) {
           max_idx_.Reshape(out_put_shape);
      }
      // If stochastic pooling, we will initialize the random index part.
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_STOCHASTIC) {
        rand_idx_.Reshape(out_put_shape);
      }


}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape2D(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
        CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
            << "corresponding to (num, channels, height, width)";
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        if (global_pooling_) {
          kernel_h_ = bottom[0]->height();
          kernel_w_ = bottom[0]->width();
        }
        pooled_height_ = static_cast<int>(ceil(static_cast<float>(
            height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
        pooled_width_ = static_cast<int>(ceil(static_cast<float>(
            width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
        if (pad_h_ || pad_w_) {
          // If we have padding, ensure that the last pooling starts strictly
          // inside the image (instead of at the padding); otherwise clip the last.
          if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
            --pooled_height_;
          }
          if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
            --pooled_width_;
          }
          CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
          CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
        }
        top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
        if (top.size() > 1) {
          top[1]->ReshapeLike(*top[0]);
        }
        // If max pooling, we will initialize the vector index part.
        if (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_MAX && top.size() == 1) {
          max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
              pooled_width_);
        }
        // If stochastic pooling, we will initialize the random index part.
        if (this->layer_param_.pooling_param().pool() ==
            PoolingParameter_PoolMethod_STOCHASTIC) {
          rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
            pooled_width_);
        }

}


// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    LOG(INFO)<<"start forward "<< this->layer_param_.name();
    if(num_spatial_axes_ ==2)
      Forward_cpu_2D(bottom, top);
    else
      Forward_cpu_ND(bottom, top);
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu_ND(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top){
  Forward_gpu(bottom, top);
  return;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  const vector<int>& bottom_shape =bottom[0]->shape();
  int num =bottom_shape[0];
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  const int* pooled_d_shape_data =pooled_d_shape_.cpu_data();
  const int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  const int* stride_shape_data = stride_shape_.cpu_data();
  const int* pad_shape_data = pad_shape_.cpu_data();
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
  //  int pooled_size =pooled_shape
   for (int n = 0; n < num; ++n) {
     for (int c = 0; c < channels_; ++c) {
       for(int p=0;p<pooled_data_length_;++p){
           vector<int> nd_point;
           vector<int>::iterator it;
           size_t pre_aixs_len =0;
           int data_axis_idx =0;
           // compute corresponding coords in nd space；
             for(int dim =num_spatial_axes_-1; dim>0; --dim){
                 if(dim==num_spatial_axes_-1){
                    data_axis_idx=p%pooled_d_shape_data[dim];
                    pre_aixs_len=pooled_d_shape_data[dim];
                 }else{
                     data_axis_idx=(p/pre_aixs_len)%pooled_d_shape_data[dim];
                     pre_aixs_len*=pooled_d_shape_data[dim];
                 }
                 it =nd_point.begin();
                 nd_point.insert(it, data_axis_idx);
               }
               data_axis_idx=(p/pre_aixs_len);
               it =nd_point.begin();
               nd_point.insert(it, data_axis_idx);
          // compute the bottom pooling coordinate range for each output
               vector<int> start_aixs_idx, end_aixs_idx;
               int total_kernel_size  =1;
              // std::cout<<"---"<<std::endl;
               for(int i =0;i<num_spatial_axes_;++i){
                // std::cout<<"nd_point ["<<i<<"] = "<<nd_point[i] <<std::endl;
                 start_aixs_idx.push_back(nd_point[i]*stride_shape_data[i]-pad_shape_data[i]);
                 end_aixs_idx.push_back(min(start_aixs_idx[i] + kernel_shape_data[i],
                                           bottom_shape[first_spatial_axis_+i]));
                 start_aixs_idx[i]=max(start_aixs_idx[i],0);

                 total_kernel_size *=(end_aixs_idx[i]-start_aixs_idx[i]);
                // std::cout<<"start_idx ["<<i<<"] = "<<start_aixs_idx[i] <<std::endl;
                 //std::cout<<"end_idx ["<<i<<"] = "<<end_aixs_idx[i] <<std::endl;
               }

              for(int i=0;i<total_kernel_size;++i)
              {
                vector<int> bottom_k_points;
                int kernel_dim =start_aixs_idx.size();
                //compute coordinate of point in the bottom kernel.
                for(int n =kernel_dim-1; n>0; --n){
                   int pooled_kernel_size =end_aixs_idx[n]-start_aixs_idx[n];
                    if(n==kernel_dim-1){
                       data_axis_idx=p%(pooled_kernel_size)+start_aixs_idx[n];
                       pre_aixs_len=pooled_kernel_size;
                    }else{
                        data_axis_idx=(i/pre_aixs_len)%pooled_kernel_size+start_aixs_idx[n];
                        pre_aixs_len*=pooled_kernel_size;
                    }
                    it =bottom_k_points.begin();
                    bottom_k_points.insert(it, data_axis_idx);
                  }
                  data_axis_idx=(i/pre_aixs_len);
                  it =bottom_k_points.begin();
                  bottom_k_points.insert(it, data_axis_idx);
           // convert coordinate to index in the bottom;

                  int data_idx=0;
                   for (int n=0;n<bottom_k_points.size();++n){
                        if(n==0){
                          data_idx = bottom_k_points[n];
                        }else{
                          data_idx*=bottom_shape[first_spatial_axis_+n];
                          data_idx+=bottom_k_points[n];
                        }
                   }

                   if (bottom_data[data_idx] > top_data[p]) {
                     top_data[p] = bottom_data[data_idx];
                     if (use_top_mask) {
                       top_mask[p] = static_cast<Dtype>(data_idx);
                     } else {
                       mask[p] = data_idx;
                     }
                   }
               }
       }
       // compute offset
       vector<int> offset(2,0);
       offset[0]=0;
       offset[1]=1;
       //offset[2]=1;
       bottom_data += bottom[0]->offset(offset);
       top_data += top[0]->offset(offset);
       if (use_top_mask) {
         top_mask += top[0]->offset(offset);
       } else {
         mask += top[0]->offset(offset);
       }
     }
   }
  break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu_2D(const vector<Blob<Dtype>*>& bottom,
const vector<Blob<Dtype>*>& top){
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}




template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
        if (!propagate_down[0]) {
          return;
        }
        if(num_spatial_axes_ ==2)
          Backward_cpu_2D(top, propagate_down, bottom);
        else
          Backward_cpu_ND(top, propagate_down, bottom);
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu_ND(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        const Dtype* top_diff = top[0]->cpu_diff();
          Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
          // Different pooling methods. We explicitly do the switch outside the for
          // loop to save time, although this results in more codes.
          caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
          // We'll output the mask to top[1] if it's of size >1.
          const bool use_top_mask = top.size() > 1;
          const int* mask = NULL;  // suppress warnings about uninitialized variables
          const Dtype* top_mask = NULL;

          const int * pooled_d_shape=pooled_d_shape_.cpu_data();
          int pooled_height_ =pooled_d_shape[0];
          int pooled_width_ =pooled_d_shape[1];
          int pooled_depth_ =pooled_d_shape[2];
          //std::cout<<"p_h ="<<pooled_height_<<std::endl;
        //  std::cout<<"p_w ="<<pooled_width_<<std::endl;
          //std::cout<<"p_d ="<<pooled_depth_<<std::endl;

          switch (this->layer_param_.pooling_param().pool()) {
          case PoolingParameter_PoolMethod_MAX:
            // The main loop
            if (use_top_mask) {
              top_mask = top[1]->cpu_data();
            } else {
              mask = max_idx_.cpu_data();
            }


            for (int n = 0; n < top[0]->num(); ++n) {
              for (int c = 0; c < channels_; ++c) {
                for (int ph = 0; ph < pooled_height_; ++ph) {
                  for (int pw = 0; pw < pooled_width_; ++pw) {
                			for (int pd = 0; pd < pooled_depth_; ++pd) {
                				const int index = (ph * pooled_width_ + pw) * pooled_depth_ + pd;
                				const int bottom_index =
                					use_top_mask ? top_mask[index] : mask[index];
                				   bottom_diff[bottom_index] += top_diff[index];
                          // std::cout<<"top index ="<<index<<std::endl;
                           //std::cout<<"diff["<<bottom_index <<"]"<<	bottom_diff[bottom_index] <<std::endl;
                				}
        			      }
                  }
                vector<int> offset_v;
                offset_v.push_back(0);
                offset_v.push_back(1);
                bottom_diff += bottom[0]->offset(offset_v);
                top_diff += top[0]->offset(offset_v);
                //std::cout<<"bottom[0]->offset(offset_v)  =" <<bottom[0]->offset(offset_v)<<std::endl;
                if (use_top_mask) {
                  top_mask += top[0]->offset(offset_v);
                } else {
                  mask += top[0]->offset(offset_v);
                }
              }
            }
            break;
        case PoolingParameter_PoolMethod_AVE:
               Backward_gpu(top,propagate_down, bottom);
          break;
        case PoolingParameter_PoolMethod_STOCHASTIC:
            NOT_IMPLEMENTED;
            break;
          default:
            LOG(FATAL) << "Unknown pooling method.";
          }


}
template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu_2D(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
