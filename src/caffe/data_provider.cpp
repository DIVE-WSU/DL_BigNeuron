#include "caffe/data_provider.hpp"
namespace caffe {

template <typename Dtype>
Data_provider<Dtype>::Data_provider(const DataProviderParameter& param)
:param_(param)
{
  batch_size_ = param_.batch_size();
}

template <typename Dtype>
Data_DB_provider<Dtype>::Data_DB_provider(const DataProviderParameter& param)
:Data_provider<Dtype>(param){

}
template <typename Dtype>
 Data_HDF5_provider<Dtype>:: Data_HDF5_provider(const DataProviderParameter& param)
:Data_provider<Dtype>(param){

  for (int i=0;i<this->batch_size_;++i)
  {this->source_data_label_pair_.push_back(new  Batch_data<Dtype>);}
  //this->source_data_label_pair_.resize(this->batch_size_);
  //hdf_blobs_.resize(batch_size_);
  // Read the source to parse the filenames.
  bool has_hdf5_source = this->param_.has_data_source();
  CHECK(has_hdf5_source)<<"hdf5 data must have a source file ...";
  const string& hdf5_source = this->param_.data_source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << hdf5_source;
  hdf5_filenames_.clear();
  std::ifstream source_file(hdf5_source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf5_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << hdf5_source;
  }
  source_file.close();
  num_files_ = hdf5_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << hdf5_source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->param_.hdf5_file_shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
 // LOG(INFO) << "Loading hdf5 ...  " << hdf5_filenames_[file_permutation_[current_file_]];
  //this->LoadHDF5FileData(hdf5_filenames_[file_permutation_[current_file_]].c_str(),0);
  //this->Load_next_batch();
  //this->Load_next_batch();
  //current_row_ = 0;

  //void loadHDF5FileData(const char* filename, int blob_idx);

}

template <typename Dtype>
void Data_HDF5_provider<Dtype>::LoadHDF5FileData(const char* filename, int blob_idx)
{
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  //int top_size = this->layer_param_.top_size();
  // we assume that each hdf5 file contains one "data" and one "label"
  //Dtype a =Dtype(1);
  vector<string> data_set_names;
  data_set_names.push_back("data");
  data_set_names.push_back("label");
  //int num_dataset =data_set_names.size();
  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  //for (int i = 0; i < data_set_names.size(); ++i) {
    //source_data_label_pair_[blob_idx].data = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    //LOG(INFO) << "hdf5_load_nd_dataset ...  " << data_set_names[0];
    hdf5_load_nd_dataset(file_id, data_set_names[0].c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, this->source_data_label_pair_[blob_idx]->data_.get());
    //LOG(INFO) << "hdf5_load_nd_dataset ...  " << data_set_names[1];
    //source_data_label_pair_[blob_idx].label = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
        hdf5_load_nd_dataset(file_id, data_set_names[1].c_str(),
            MIN_DATA_DIM, MAX_DATA_DIM, this->source_data_label_pair_[blob_idx]->label_.get());


  //}
   //LOG(INFO) << "finished loading hdf5 file ";
  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(this->source_data_label_pair_[blob_idx]->data_->num_axes(), 1) << "Input must have at least 1 axis.";
  vector<int> d_shape =this->source_data_label_pair_[blob_idx]->data_->shape();
  vector<int> l_shape =this->source_data_label_pair_[blob_idx]->label_->shape();
  LOG(INFO)<<"d_size =" <<d_shape.size();

  for (int i=0;i<d_shape.size();++i)
     LOG(INFO)<<"loaded data shape : " <<d_shape[i];

  LOG(INFO)<<" ";
  for (int i=0;i<l_shape.size();++i)
        LOG(INFO)<<"loaded label shape : " <<l_shape[i];

 //make data blob to have 2 extra dimention ; num and channel but all ==1

  if(d_shape.size()==3){
    //there is no num and channel in data , so appending num and channel to the data
    d_shape.insert(d_shape.begin(),1);
    d_shape.insert(d_shape.begin(),1);
    this->source_data_label_pair_[blob_idx]->data_->Reshape(d_shape);
    //LOG(INFO)<<"Reshapeing the source data blob : adding num 1 and channel 1 : ";
    //vector<int>::iterator it=
    //this->source_data_label_pair_[blob_idx].data_
  }else if(d_shape.size()==4){
    d_shape.insert(d_shape.begin(),1);
    this->source_data_label_pair_[blob_idx]->data_->Reshape(d_shape);
  }


  int diff_d_l=d_shape.size()-l_shape.size();
  CHECK_GE(diff_d_l,0);
  if(diff_d_l>0){
      for(int i=0;i<diff_d_l;++i)
      {l_shape.insert(l_shape.begin(),1);}
      this->source_data_label_pair_[blob_idx]->label_->Reshape(l_shape);
  }


  d_shape =this->source_data_label_pair_[blob_idx]->data_->shape();
  l_shape =this->source_data_label_pair_[blob_idx]->label_->shape();
  LOG(INFO)<<"d_size =" <<d_shape.size();

  for (int i=0;i<d_shape.size();++i)
     LOG(INFO)<<"data shape after prependig : " <<d_shape[i];

  LOG(INFO)<<" ";
  for (int i=0;i<l_shape.size();++i)
        LOG(INFO)<<"label shape after prependig  : " <<l_shape[i];


  //const int num = this->source_data_label_pair_[blob_idx].data_->shape(0);
  //CHECK_EQ(num,1);// for patching we assumed that very hdf file has onely one large 3D valume.
  // for (int i = 1; i < num_dataset; ++i) {
  //   CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  // }
  // Default to identity permutation.
  // data_permutation_.clear();
  // data_permutation_.resize(hdf_blobs_[0]->shape(0));
  // for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
  //   data_permutation_[i] = i;
  //
  // // Shuffle if needed.
  // if (this->param_.data_shuffle()) {
  //   std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
  //   DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0)
  //              << " rows (shuffled)";
  // } else {
  //   DLOG(INFO) << "Successully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  // }


  LOG(INFO) << "loaded hdf5 file " << filename;
}

template <typename Dtype>
 void Data_HDF5_provider<Dtype>::Load_next_batch(int data_idx){
    //LOG(INFO) <<"loading data index  = "<<numData;
  // for(int i=0;i<numData;++i){
     CHECK_GE(data_idx,0);
     CHECK_LE(data_idx, this->batch_size_-1);
     if(current_file_ >=num_files_) {current_file_  =0;}
     //LOG(INFO) <<"loading HDF5 file : "<<"'"<<hdf5_filenames_[file_permutation_[current_file_]]<<"'";
     this->LoadHDF5FileData(hdf5_filenames_[file_permutation_[current_file_]].c_str(), data_idx);
     current_file_++;
  //  }
 }

 template <typename Dtype>
  void Data_HDF5_provider<Dtype>::Load_next_batch(){
  //   LOG(INFO) <<"loading next batch hdf5 files = "<<numData;
    for(int i=0;i<this->batch_size_;++i){
        Load_next_batch(i);
      // if(current_file_ >=num_files_) {current_file_  =0;}
      //   LOG(INFO) <<"loading file  = "<<hdf5_filenames_[file_permutation_[current_file_]];
      //   this->LoadHDF5FileData(hdf5_filenames_[file_permutation_[current_file_]].c_str(), i);
      //   current_file_++;
     }
  }


 INSTANTIATE_CLASS(Batch_data);
 INSTANTIATE_CLASS(Data_HDF5_provider);
 INSTANTIATE_CLASS(Data_DB_provider);
 INSTANTIATE_CLASS(Data_provider);

 }
