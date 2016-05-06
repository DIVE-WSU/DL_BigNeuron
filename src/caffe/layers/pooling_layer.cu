#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype, int num_axes>
__global__ void MaxPoolForward_ND(const int nthreads,
  const Dtype* const bottom_data, const int num, const int channels,
  const int* bottom_shape, const int* top_shape, const int* kernel,
  const int* stride,const int* pad,
  Dtype* const top_data, int*  mask, Dtype* top_mask){
    int d_top_pt[num_axes];  // NOLINT(runtime/arrays)
    //int d_iter[num_axes];  // NOLINT(runtime/arrays)
    int d_start_pt[num_axes];
    int d_end_pt[num_axes];
    int d_bottom_pt[num_axes];
    int i,j;
    CUDA_KERNEL_LOOP(index, nthreads) {
      int channel_in = index;
      int channel_out = 1;
      int n  =0;
      int c =0;
      int kernel_length =1;
      for (i = num_axes - 1; i >= 0; --i) {
        d_top_pt[i] = channel_in % top_shape[i];
        channel_in /= top_shape[i];
        channel_out *= kernel[i];
       }
        c = channel_in%channels;
        n =channel_in/channels;
      for(i=0;i<num_axes;++i){
       kernel_length*=kernel[i];
      }

      int nc_len =(n * channels + c);
      int bottom_dim_len=1;
      for(i=0;i<num_axes;++i){
        bottom_dim_len*=bottom_shape[i];
      }

      Dtype maxval = -FLT_MAX;
      int maxidx = -1;
      const Dtype* const bottom_slice =
           bottom_data + nc_len * bottom_dim_len;
     int  cur_total_k_size =1;
     for(i=0;i<num_axes;++i){
       d_start_pt[i]=(d_top_pt[i]*stride[i]-pad[i]);
       d_end_pt[i]=(min(d_start_pt[i] + kernel[i],
                                 bottom_shape[i]));
       d_start_pt[i]=max(d_start_pt[i],0);
       cur_total_k_size *=(d_end_pt[i]-d_start_pt[i]);
     }

     for(i=0;i<cur_total_k_size; ++i){
       //compute coordinate of point in the bottom kernel.
       int inner = i;
       for( j=num_axes-1; j>=0; --j){
            int pooled_kernel_size =d_end_pt[j]-d_start_pt[j];
            d_bottom_pt[j]=inner%pooled_kernel_size+d_start_pt[j];
            inner /=pooled_kernel_size;
          }
       int data_idx =0;
       for(j=0; j<num_axes; ++j){
         if(j==0){
           data_idx = d_bottom_pt[j];
         }else{
           data_idx*=bottom_shape[j];
           data_idx+=d_bottom_pt[j];
         }
       }
       if (bottom_slice[ data_idx] > maxval) {
         maxidx =  data_idx;
         maxval = bottom_slice[maxidx];
       }
     }
     top_data[index] = maxval;
     if (mask) {
       mask[index] = maxidx;
     } else {
       top_mask[index] = maxidx;
     }

  }
}


template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}


template <typename Dtype, int num_axes>
__global__ void AvePoolForward_ND(const int nthreads,
  const Dtype* const bottom_data, const int num, const int channels,
  const int* bottom_shape, const int* top_shape, const int* kernel,
  const int* stride,const int* pad,
  Dtype* const top_data, int*  mask, Dtype* top_mask){
    int d_top_pt[num_axes];  // NOLINT(runtime/arrays)
    int d_start_pt[num_axes];
    int d_end_pt[num_axes];
    int d_bottom_pt[num_axes];
    int i,j;
    CUDA_KERNEL_LOOP(index, nthreads) {
      int channel_in = index;
      int channel_out = 1;
      int n  =0;
      int c =0;
      int kernel_length =1;
      for (i = num_axes - 1; i >= 0; --i) {
        d_top_pt[i] = channel_in % top_shape[i];
        channel_in /= top_shape[i];
        channel_out *= kernel[i];
       }
        c = channel_in%channels;
        n =channel_in/channels;
      for(i=0;i<num_axes;++i){
       kernel_length*=kernel[i];
      }

      int nc_len =(n * channels + c);
      int bottom_dim_len=1;
      for(i=0;i<num_axes;++i){
        bottom_dim_len*=bottom_shape[i];
      }
      const Dtype* const bottom_slice =
           bottom_data + nc_len * bottom_dim_len;
       int  cur_total_k_size =1;
       for(i=0;i<num_axes;++i){
         d_start_pt[i]=(d_top_pt[i]*stride[i]-pad[i]);
         d_end_pt[i]=(min(d_start_pt[i] + kernel[i],
                                   bottom_shape[i]));
         d_start_pt[i]=max(d_start_pt[i],0);
         cur_total_k_size *=(d_end_pt[i]-d_start_pt[i]);
       }
       Dtype aveval = 0;
       for(i=0;i<cur_total_k_size; ++i){
         //compute coordinate of point in the bottom kernel.
         int inner = i;
         for( j=num_axes-1; j>=0; --j){
              int pooled_kernel_size =d_end_pt[j]-d_start_pt[j];
              d_bottom_pt[j]=inner%pooled_kernel_size+d_start_pt[j];
              inner /=pooled_kernel_size;
            }
           int data_idx =d_bottom_pt[0];
           for(j=1; j<num_axes; ++j){
               data_idx*=bottom_shape[j];
               data_idx+=d_bottom_pt[j];
           }
           aveval +=  bottom_slice[data_idx];
        }
       top_data[index] = aveval/cur_total_k_size;
    }
  }



template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
        if(this->layer_param_.phase()==PREDICT)
          {LOG(INFO)<<"start forward "<< this->layer_param_.name();}
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  const int* kernel=NULL;//kernel_shape_.gpu_data();
  const int* stride=NULL;//stride_shape_.gpu_data();
  const int* top_shape=NULL;//pooled_d_shape_.gpu_data();
  const int* bottom_shape=NULL;//bottom_d_shape_.gpu_data();
  const int* pad=NULL;//pad_shape_.gpu_data();
 if(num_spatial_axes_!=2){
  kernel=kernel_shape_.gpu_data();
  stride=stride_shape_.gpu_data();
  top_shape=pooled_d_shape_.gpu_data();
  bottom_shape=bottom_d_shape_.gpu_data();
  pad=pad_shape_.gpu_data();
}

  //LOG(INFO)<<"start gpu pooling forwarding...";
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    if(num_spatial_axes_ ==2){
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    }else{


      // NOLINT_NEXT_LINE(whitespace/operators)
      //LOG(INFO)<<"start forword ND pooling";
      switch(num_spatial_axes_){
        case 1:
        MaxPoolForward_ND<Dtype,1 ><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
          bottom_data,  bottom[0]->num(), channels_,
          bottom_shape, top_shape, kernel,
          stride,pad,top_data,mask,top_mask);
        break;
        case 2:
        MaxPoolForward_ND<Dtype,2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
          bottom_data,  bottom[0]->num(), channels_,
          bottom_shape, top_shape, kernel,
          stride,pad,top_data,mask,top_mask);
        break;
        case 3:
        MaxPoolForward_ND<Dtype,3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
          bottom_data,  bottom[0]->num(), channels_,
          bottom_shape, top_shape, kernel,
          stride,pad,top_data,mask,top_mask);
        break;
        case 4:
        MaxPoolForward_ND<Dtype,4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
          bottom_data,  bottom[0]->num(), channels_,
          bottom_shape, top_shape, kernel,
          stride,pad,top_data,mask,top_mask);
        break;
        case 5:
        MaxPoolForward_ND<Dtype,5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
          bottom_data,  bottom[0]->num(), channels_,
          bottom_shape, top_shape, kernel,
          stride,pad,top_data,mask,top_mask);
        break;
        default:
        LOG(FATAL) << "Unsupported pooling dimension.";
      }
      //LOG(INFO)<<"end forword ND pooling";

    }
    //  LOG(INFO)<<"end gpu pooling forwarding...";
    break;
  case PoolingParameter_PoolMethod_AVE:
      if(num_spatial_axes_ ==2){
          // NOLINT_NEXT_LINE(whitespace/operators)
          AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
              count, bottom_data, bottom[0]->num(), channels_,
              height_, width_, pooled_height_, pooled_width_, kernel_h_,
              kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
      }else{
        switch(num_spatial_axes_){
        case 1:
          AvePoolForward_ND<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            bottom_data,  bottom[0]->num(), channels_,
            bottom_shape, top_shape, kernel,
            stride,pad,top_data,mask,top_mask);
            break;
        case 2:
          AvePoolForward_ND<Dtype,2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            bottom_data,  bottom[0]->num(), channels_,
            bottom_shape, top_shape, kernel,
            stride,pad,top_data,mask,top_mask);
          break;
        case 3:
        AvePoolForward_ND<Dtype,3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
          bottom_data,  bottom[0]->num(), channels_,
          bottom_shape, top_shape, kernel,
          stride,pad,top_data,mask,top_mask);
            break;
        case 4:
          AvePoolForward_ND<Dtype,4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
            bottom_data,  bottom[0]->num(), channels_,
            bottom_shape, top_shape, kernel,
            stride,pad,top_data,mask,top_mask);
            break;
        default:
        LOG(FATAL) << "Unsupported pooling dimension.";
       }

      }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  if(this->layer_param_.phase()==PREDICT){
    LOG(INFO)<<"Free GPU mem...";
    bottom[0]->data()->free();
    bottom[0]->diff()->free();
  }
  CUDA_POST_KERNEL_CHECK;

}



template <typename Dtype, int num_axes>
__global__ void MaxPoolBackward_ND(const int nthreads, const Dtype*  top_diff,
    const int*  mask, const Dtype* top_mask, const int num,
    const int channels, const int* bottom_shape,
    const int* top_shape, const int* kernel, const int* stride, const int* pad,
    Dtype* const bottom_diff){
      //int d_top_pt[num_axes];  // NOLINT(runtime/arrays)
      //int d_iter[num_axes];  // NOLINT(runtime/arrays)
      //int i;
    CUDA_KERNEL_LOOP(index, nthreads) {
      int height=bottom_shape[0];
      int width= bottom_shape[1];
      int depth =bottom_shape[2];
      int kernel_h=kernel[0];
      int kernel_w=kernel[1];
      int kernel_d=kernel[2];
      int pad_h=pad[0];
      int pad_w=pad[1];
      int pad_d=pad[2];
      int pooled_height =top_shape[0];
      int pooled_width =top_shape[1];
      int pooled_depth =top_shape[2];
      int stride_h =stride[0];
      int stride_w =stride[1];
      int stride_d =stride[2];
  int d = index % depth;
  int w = (index / depth) % width;
  int h = (index / depth / width) % height;
  int c = (index / depth / width / height) % channels;
  int n = index / depth / width / height / channels;
  int phstart =
      (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
  int phend = min((h + pad_h) / stride_h + 1, pooled_height);
  int pwstart =
      (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
  int pwend = min((w + pad_w) / stride_w + 1, pooled_width);

int pdstart =
      (d + pad_d < kernel_d) ? 0 : (d + pad_d - kernel_d) / stride_d + 1;
  int pdend = min((d + pad_d) / stride_d + 1, pooled_depth);
//
  Dtype gradient = 0;
  int offset = (n * channels + c) * pooled_height * pooled_width * pooled_depth;
  top_diff += offset;
  if (mask) {
    mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int pd = pdstart; pd < pdend; ++pd) {
          if (mask[(ph * pooled_width + pw) * pooled_depth + pd] == (h * width + w) * depth +d) {
            gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd];
          }
        }
      }
    }
  } else {
    top_mask += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for (int pd = pdstart; pd < pdend; ++pd){
          if (top_mask[(ph * pooled_width + pw) * pooled_depth + pd] == (h * width + w) * depth +d) {
              gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd];
            }
          }
      }
    }
  }
  bottom_diff[index] = gradient;





    //   int n  =0;
    //   int c =0;
    //   //int kernel_length =1;
    //
    //   //d_temp[num_axes-1]=channel_in%top_shape[num_axes-1];
    //   //channel_out*=kernel_shape[num_axes-1]
    //
    //   //int channel_in = index;
    //   //int channel_out = 1;
    //   // for (i = num_axes - 1; i >= 0; --i) {
    //   //   d_top_pt[i] = channel_in % top_shape[i];
    //   //   channel_in /= top_shape[i];
    //   //   channel_out *= kernel[i];
    //   //  }
    //   //   c = channel_in%channels;
    //   //   n =channel_in/channels;
    //   // int top_shape_len=1;
    //   // for(i=0;i<num_axes;++i){
    //   //   top_shape_len*=top_shape[i];
    //   // }
    //
    //   int bottom_shape_len=1;
    //   for(i=0;i<num_axes;++i){
    //     bottom_shape_len*=bottom_shape[i];
    //   }
    //
    //   const int offset = (n * channels + c) * bottom_shape_len;
    //   Dtype* bottom_slice =  bottom_diff+offset;
    //
    // //  bottom_diff[index] = gradient;
    //   if (mask){
    //       bottom_slice[mask[index]]+=top_diff[index];
    //   }else{
    //      int idx =(int)top_mask[index];
    //       bottom_slice[idx]+=top_diff[index];
    //   }

    }
  }


  template <typename Dtype, int num_axes>
  __global__ void AvePoolBackward_ND(const int nthreads, const Dtype*  top_diff,
      const int*  mask, const Dtype* top_mask, const int num,
      const int channels, const int* bottom_shape,
      const int* top_shape, const int* kernel, const int* stride, const int* pad,
      Dtype* const bottom_diff){
        //int d_top_pt[num_axes];  // NOLINT(runtime/arrays)
        //int d_iter[num_axes];  // NOLINT(runtime/arrays)
        //int i;
      CUDA_KERNEL_LOOP(index, nthreads) {
        int depth =bottom_shape[2];
        int width= bottom_shape[1];
        int height=bottom_shape[0];
        int kernel_h=kernel[0];
        int kernel_w=kernel[1];
        int kernel_d=kernel[2];
        int pad_h=pad[0];
        int pad_w=pad[1];
        int pad_d=pad[2];
        int pooled_height =top_shape[0];
        int pooled_width =top_shape[1];
        int pooled_depth =top_shape[2];
        int stride_h =stride[0];
        int stride_w =stride[1];
        int stride_d =stride[2];

        int d = index % depth +pad_d;
        int w = (index / depth) % width + pad_w;
        int h = (index / depth / width) % height + pad_h;
        int c = (index / depth / width / height) % channels;
        int n = index / depth / width / height / channels;
        int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
        int phend = min(h / stride_h + 1, pooled_height);
        int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
        int pwend = min(w / stride_w + 1, pooled_width);
    	  int pdstart = (d < kernel_d) ? 0 : (d - kernel_d) / stride_d + 1;
        int pdend = min(d / stride_d + 1, pooled_depth);

        Dtype gradient = 0;
        top_diff += (n * channels + c) * pooled_height * pooled_width * pooled_depth;
        for (int ph = phstart; ph < phend; ++ph) {
          for (int pw = pwstart; pw < pwend; ++pw) {
        	    for (int pd = pdstart; pd < pdend; ++pd) {
        			// figure out the pooling size
        			int hstart = ph * stride_h - pad_h;
        			int wstart = pw * stride_w - pad_w;
        			int dstart = pd * stride_d - pad_d;
        			int hend = min(hstart + kernel_h, height + pad_h);
        			int wend = min(wstart + kernel_w, width + pad_w);
        			int dend = min(dstart + kernel_d, depth + pad_d);
        			int pool_size = (hend - hstart) * (wend - wstart) * (dend - dstart);
        			gradient += top_diff[(ph * pooled_width + pw) * pooled_depth + pd] / pool_size;
    		      }
            }
          }
        bottom_diff[index] = gradient;
    }
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset

    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    LOG(INFO)<<"pooling backwarding. retruned..";
    return;
  }
//  LOG(INFO)<<"start gpu pooling backwarding...";
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  const int top_count =top[0]->count();

  const int* kernel_shape=NULL;//kernel_shape_.gpu_data();
  const int* stride_shape=NULL;//stride_shape_.gpu_data();
  const int* top_shape=NULL;//pooled_d_shape_.gpu_data();
  const int* bottom_shape=NULL;//bottom_d_shape_.gpu_data();
  const int* pad_shape=NULL;//pad_shape_.gpu_data();
  int num_axes =bottom[0]->num_axes();
 if(num_axes !=4){
  kernel_shape=kernel_shape_.gpu_data();
  stride_shape=stride_shape_.gpu_data();
  top_shape=pooled_d_shape_.gpu_data();
  bottom_shape=bottom_d_shape_.gpu_data();
  pad_shape=pad_shape_.gpu_data();
}

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }

    if(num_axes==4){
    // NOLINT_NEXT_LINE(whitespace/operators)
    //LOG(INFO)<<"Run 2d backward pooling";
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    }else{

       // NOLINT_NEXT_LINE(whitespace/operators)
  //     LOG(INFO)<<"start backword ND pooling";
       switch (num_spatial_axes_) {
         case 1:
         MaxPoolBackward_ND<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
             top_mask, top[0]->num(), channels_,
             bottom_shape, top_shape,kernel_shape,
             stride_shape,pad_shape, bottom_diff);
         break;
         case 2:
         MaxPoolBackward_ND<Dtype,2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             top_count, top_diff, mask,
             top_mask, top[0]->num(), channels_,
             bottom_shape, top_shape, kernel_shape,
             stride_shape,pad_shape, bottom_diff);
         break;
         case 3:
         MaxPoolBackward_ND<Dtype,3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
             top_mask, top[0]->num(), channels_,
             bottom_shape, top_shape,kernel_shape,
             stride_shape, pad_shape, bottom_diff);
         break;
         case 4:
         MaxPoolBackward_ND<Dtype,4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
             top_mask, top[0]->num(), channels_,
             bottom_shape, top_shape,kernel_shape,
             stride_shape,pad_shape, bottom_diff);
         break;
         case 5:
         MaxPoolBackward_ND<Dtype,5><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask,
             top_mask, top[0]->num(), channels_,
             bottom_shape, top_shape,kernel_shape,
             stride_shape,pad_shape, bottom_diff);
         break;
         default:
           LOG(FATAL) << "unsupported pooling dimension.";
       }

    //      LOG(INFO)<<"end gpu pooling backwarding...";
    }

    break;
  case PoolingParameter_PoolMethod_AVE:
      if(num_axes==4){
        // NOLINT_NEXT_LINE(whitespace/operators)
        AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->num(), channels_,
            height_, width_, pooled_height_, pooled_width_, kernel_h_,
            kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
      }else{
        switch (num_spatial_axes_) {
          case 1:
          AvePoolBackward_ND<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
              top_mask, top[0]->num(), channels_,
              bottom_shape, top_shape,kernel_shape,
              stride_shape,pad_shape, bottom_diff);
              break;
          case 2:
          AvePoolBackward_ND<Dtype,2><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
              top_mask, top[0]->num(), channels_,
              bottom_shape, top_shape,kernel_shape,
              stride_shape,pad_shape, bottom_diff);
              break;
          case 3:
          AvePoolBackward_ND<Dtype,3><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
              top_mask, top[0]->num(), channels_,
              bottom_shape, top_shape,kernel_shape,
              stride_shape,pad_shape, bottom_diff);
              break;
          case 4:
          AvePoolBackward_ND<Dtype,4><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
             count, top_diff, mask,
              top_mask, top[0]->num(), channels_,
              bottom_shape, top_shape,kernel_shape,
              stride_shape,pad_shape, bottom_diff);
              break;
          default:
            LOG(FATAL) << "unsupported pooling dimension.";
          }
        }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
