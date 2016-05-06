#ifndef CAFFE_DATA_PROVIDER_HPP_
#define CAFFE_DATA_PROVIDER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
//#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template <typename Dtype> class Data_DB_provider;
template <typename Dtype> class Data_HDF5_provider;

template <typename Dtype>
class Batch_data {
   public:
    Batch_data()
    { 
	  data_.reset(new Blob<Dtype>);
      label_.reset(new Blob<Dtype>);
	  //LOG(INFO)<<"rest new batch_data";
	}
    shared_ptr<Blob<Dtype> > data_, label_;
  };

template <typename Dtype>
class Data_provider{
public:
  explicit Data_provider(const DataProviderParameter& param);
  ~Data_provider(){};
  virtual void Load_next_batch(int data_blob_idx)=0;
  virtual void Load_next_batch()=0;
  inline int get_current_batch_size(){return batch_size_;};
  inline const Batch_data<Dtype>& getOneData(int idx){ return *source_data_label_pair_[idx];};
protected:
  DataProviderParameter param_;
  int batch_size_;
  vector< Batch_data<Dtype>* > source_data_label_pair_;
};



template <typename Dtype>
class Data_DB_provider:public Data_provider<Dtype>{
public:
  explicit Data_DB_provider(const DataProviderParameter& param);
  ~Data_DB_provider(){};
  virtual void Load_next_batch(int data_blob_idx){};
  virtual void Load_next_batch(){};
};

template <typename Dtype>
class Data_HDF5_provider:public Data_provider<Dtype>{
public:
  explicit Data_HDF5_provider(const DataProviderParameter& param);
  ~Data_HDF5_provider(){};
  virtual void Load_next_batch(int data_blob_idx);
  virtual void Load_next_batch();
protected:
  void LoadHDF5FileData(const char* filename, int blob_idx);
  std::vector<std::string> hdf5_filenames_;
  unsigned int num_files_;
  unsigned int current_file_;
  std::vector<shared_ptr<Blob<Dtype> > > hdf_blobs_;
  std::vector<unsigned int> data_permutation_;
  std::vector<unsigned int> file_permutation_;
  };


  template <typename Dtype>
  static shared_ptr<Data_provider<Dtype> > Make_data_provider_instance(const DataProviderParameter& data_provider_param)
  {
     static DataProviderParameter_DS ds = data_provider_param.backend();
     if(ds==DataProviderParameter_DS_HDF5){
        return shared_ptr<Data_provider<Dtype> > (new Data_HDF5_provider<Dtype>(data_provider_param));
      }
    else {//if(ds==DataProviderParameter_DS_LMDB || ds==DataProviderParameter_DS_LEVELDB){
       return shared_ptr<Data_provider<Dtype> > (new Data_DB_provider<Dtype>(data_provider_param));
     }
  };

}
#endif
