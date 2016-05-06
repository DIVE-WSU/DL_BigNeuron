#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include "caffe/util/io.hpp"
#include "caffe/common.hpp"
#include "caffe/patch_sampler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"


namespace caffe {


//template class PatchSampler<float>;
//template class PatchSampler<double>;

using boost::weak_ptr;

template <typename Dtype>
map<const string, weak_ptr<Runner<Dtype> > > PatchSampler<Dtype>::runners_;

static boost::mutex runners_mutex_;
static int count_patch_instances = 0;
template <typename Dtype>
PatchSampler<Dtype>::PatchSampler(const LayerParameter& param)
    :param_(param),
    queue_pair_(new QueuePair_Batch<Dtype>(param)),
    d_provider_(Make_data_provider_instance<Dtype>(param.data_provider_param())),
    patch_coord_finder_(new PatchCoordFinder<Dtype>(param)),
    sample_selector_(new SampleSelector<Dtype>(param)),
    data_transformer_nd(new DataTransformerND<Dtype>(param.transform_nd_param()))
{
  // Get or create a body
  //----------------------------------------------------------------------------------------------------------
  // Change code not to be sequential as in dataReader, instead
  // we allow or mutilple runner to run in paraller(random patching from the loaded data sets)
  //------------------------------------------------------------------------------------------------------------
  /*boost::mutex::scoped_lock lock(runners_mutex_);
  string key = source_key(param);
  LOG(INFO)<<"patch sample key = "<<key;
  weak_ptr<Runner<Dtype> >& weak = runners_[key];
  runner_ = weak.lock();
  count_patch_instances = count_patch_instances+1;
  if (!runner_) {
    runner_.reset(new Runner<Dtype>(param,*this));
    runners_[key] = weak_ptr<Runner<Dtype> >(runner_);
    LOG(INFO)<<"reset runner ...";
  }
  runner_->new_queue_pairs_.push(queue_pair_); */

  runner_.reset(new Runner<Dtype>(param,*this));
  //runner_ = new Runner<Dtype>(param,*this);
  runner_->new_queue_pairs_.push(queue_pair_);


  LOG(INFO)<<"runner setup done ... count =" <<count_patch_instances;
  //count_patch_instances++；
  patches_per_data_batch_ = param_.patch_sampler_param().patches_per_data_batch();
  data_transformer_nd->InitRand();
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  patch_count_ =0;
  int data_patch_axims  = param_.patch_sampler_param().data_patch_shape().dim_size();
  int label_patch_axims = param_.patch_sampler_param().label_patch_shape().dim_size();
  CHECK_EQ(data_patch_axims,label_patch_axims);
  for(int i =0;i<data_patch_axims;++i){
    patch_data_shape_.push_back(param_.patch_sampler_param().data_patch_shape().dim(i));
    patch_label_shape_.push_back(param_.patch_sampler_param().label_patch_shape().dim(i));
    CHECK_GE(patch_data_shape_[i],patch_label_shape_[i]);
  }

}

template <typename Dtype>
PatchSampler<Dtype>::~PatchSampler() {
  /*string key = source_key(runner_->param_);
  runner_.reset();
  boost::mutex::scoped_lock lock(runners_mutex_);
  if (runners_[key].expired()) {
    runners_.erase(key);
  }*/
  runner_.reset();
}

template <typename Dtype>
unsigned int PatchSampler<Dtype>::PrefetchRand(){
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
vector<int> PatchSampler<Dtype>::patch_data_shape(){
  if (dest_data_shape_.size()==0){
    //LOG(INFO)<<"start readone";
    ReadOnePatch(queue_pair_.get());
    //LOG(INFO)<<"finished readone";
  }
//  LOG(INFO)<<"dest_data_shape_ size = "<<dest_data_shape_.size();
  return dest_data_shape_;
}
template <typename Dtype>
vector<int> PatchSampler<Dtype>::patch_label_shape(){
  if (dest_label_shape_.size()==0){
    ReadOnePatch(queue_pair_.get());
  }
  return dest_label_shape_;
}

template <typename Dtype>
void PatchSampler<Dtype>::ReadOnePatch(QueuePair_Batch<Dtype>* qb ){
  // load new data to memomry pool
// LOG(INFO)<<"patch read count "<<patch_count_;
  CPUTimer timer;
  count_m_mutex_.lock();
  if(patch_count_%patches_per_data_batch_ ==0)
  {
      int load_idx=PrefetchRand()% d_provider_->get_current_batch_size();
      LOG(INFO)<<"loading batch patch_count = "<<patch_count_;
      patch_count_==0?d_provider_->Load_next_batch()
                      :d_provider_->Load_next_batch(load_idx);
	  //d_provider_->Load_next_batch();
                     // :d_provider_->Load_next_batch();
      // apply mean has some logical error here, need to be corrected ?
      if(patch_count_==0){
        for(int i= 0;i<d_provider_->get_current_batch_size();++i){
          const Batch_data<Dtype> source_d_l=d_provider_->getOneData(i);
          data_transformer_nd->ApplyMean(source_d_l.data_.get(),
                                         source_d_l.data_.get());
        }
      }else{
		LOG(INFO)<<"replace index = "<<load_idx;
        const Batch_data<Dtype> source_d_l=d_provider_->getOneData(load_idx);
        data_transformer_nd->ApplyMean(source_d_l.data_.get(),
                                       source_d_l.data_.get());
      }
  }

  patch_count_++;


  int data_idx=PrefetchRand()% d_provider_->get_current_batch_size();

  const Batch_data<Dtype>& source_data_label=d_provider_->getOneData(data_idx);
  //LOG(INFO)<<"reading one patch from data blob "<<data_idx<<"  out of "<<d_provider_->get_current_batch_size();

  //LOG(INFO)<< "dong reading hd5";
  // take input patch_data then warp a patch and put it to patch_data;
  // the function that address the probability of selecting classe need to be addressed
   //CropCenterInfo<Dtype>& PeekCropCenterPoint(Blob<Dtype>* input_blob);

  //   const CropCenterInfo<Dtype>& PeekCropCenterPoint(Blob<Dtype>* input_blob);
  //const CropCenterInfo<Dtype>& crp_cent_info=data_transformer_nd->PeekCropCenterPoint(source_data_label.label_.get());

  vector<int> s_data_shape=source_data_label.label_.get()->shape();
  CHECK_GE(s_data_shape.size(),3);
  vector<int> real_data_shape(s_data_shape.begin()+2,s_data_shape.end());
  //  LOG(INFO)<<s_data_shape[0]<<":"<<s_data_shape[1]<<":"<<s_data_shape[2]<<":"<<s_data_shape[3]<<":"<<s_data_shape[4];
  patch_coord_finder_->SetInputShape(real_data_shape);

  vector<int> randPt ;
  int pt_label_value ;
  do{
      randPt                = patch_coord_finder_->GetRandomPatchCenterCoord();
      pt_label_value        = static_cast<int> (data_transformer_nd->ReadOnePoint(source_data_label.label_.get(), randPt));
  }while (!sample_selector_->AcceptGivenLabel(pt_label_value));
    //LOG(INFO)<<"x:Y:Z "<< randPt[0]<<":"<<randPt[1]<<":"<<randPt[2];
 // LOG(INFO)<<"selected Label = "<<pt_label_value;


  vector<int> data_offset     = patch_coord_finder_->GetDataOffset();
  vector<int> label_offset    = patch_coord_finder_->GetLabelOffset();
  //double  trans_time = 0;
  //timer.Start();
 // LOG(INFO)<<"size of qb->free == " <<qb->free_.size();
  Batch_data<Dtype>* patch_data_label = qb->free_.pop();
  data_transformer_nd->Transform(source_data_label.data_.get(),
                                    patch_data_label->data_.get(),
                                    data_offset,
                                    patch_data_shape_);
  data_transformer_nd->Transform(source_data_label.label_.get(),
                                    patch_data_label->label_.get(),
                                    label_offset,
                                    patch_label_shape_);
   //if(label_offset[2]<=0 || label_offset[1]<=0||label_offset[0]<=0)
	//   LOG(INFO)<<"label offset small "<<label_offset[2]<<":"<<label_offset[1]<<":"<<label_offset[0];

  //LOG(INFO)<<"end transfer";
  int new_label=(int)  patch_data_label->label_.get()->cpu_data()[0];
  if(patch_data_label->label_.get()->count()==1)
  {CHECK_EQ(new_label,pt_label_value)<<"select label must be equal to the center";}
  //LOG(INFO)<<"push to full queue";
  qb->full_.push(patch_data_label);
   const vector<int>& source_data_shape =source_data_label.data_->shape();
   const vector<int>& source_label_shape =source_data_label.label_->shape();

   count_m_mutex_.unlock();

  //LOG(INFO)<<"unlock mutex";

//  count_m_mutex_.unlock();

    // trans_time += timer.MicroSeconds();
    // timer.Stop();
    // LOG(INFO) << "     Trans time: " << trans_time / 1000 << " ms.";


  int d_dims  =source_data_shape.size();
  int l_dims  =source_label_shape.size();
  int d_num   =source_data_shape[0];
  int l_num   =source_label_shape[0];
  CHECK_EQ(d_dims,l_dims);
  CHECK_EQ(d_num,1);
  CHECK_EQ(l_num,1);
  for(int i=2;i<d_dims;i++){
    int d_dim=source_data_shape[i];
    int l_dim=source_label_shape[i];
    CHECK_EQ(d_dim,l_dim);
  }
//  dest_label_shape_.clear();
//  dest_data_shape_.clear();

  //LOG(INFO)<<"put q back to quque";
  // setting patch data and label shape;
  dest_label_shape_=patch_data_label->label_->shape();
  dest_data_shape_=patch_data_label->data_->shape();
  //LOG(INFO)<<"done reading oen patch";
}
//
template <typename Dtype>
QueuePair_Batch<Dtype>::QueuePair_Batch(const LayerParameter& param) {
  // Initialize the free queue with requested number of blobs
//  Batch_dat
  int batch_size = param.patch_sampler_param().batch_size();
//  int patch_dims = param.patch_sampler_param().patch_shape().dim_size();
  int data_patch_dims = param.patch_sampler_param().data_patch_shape().dim_size();
  int label_patch_dims = param.patch_sampler_param().label_patch_shape().dim_size();
  CHECK_GE(data_patch_dims,1);
  CHECK_EQ(data_patch_dims,label_patch_dims);
  vector<int> data_patch_shape, label_patch_shape;
  data_patch_shape.push_back(batch_size);
  label_patch_shape.push_back(batch_size);

  data_patch_shape.push_back(1);  // for channels
  label_patch_shape.push_back(1);
  //LOG(INFO)<<"batch_size = " << label_shape[0];
  for(int i=0;i<data_patch_dims;i++){
   //  int dim =param.patch_sampler_param().patch_shape().dim(i);
   int dim_d=param.patch_sampler_param().data_patch_shape().dim(i);
   int dim_l=param.patch_sampler_param().label_patch_shape().dim(i);
    data_patch_shape.push_back(dim_d);
    label_patch_shape.push_back(dim_l);
  //  LOG(INFO)<<"batch_size = " << patch_shape[i+2] <<" crop_dim = "<<dim;
  }



  for (int i = 0; i < batch_size; ++i) {
    //Batch_data<Dtype>* b_d =new
  //  Batch_data<Dtype>* b_data =new Batch_data<Dtype>();
    free_.push(new Batch_data<Dtype>);
    //Batch_data<Dtype>& b_data = *( free_.peek());
  //  Batch_data<Dtype>& b_data = *( free_.pop());

    //  b_data.data_.reset(new Blob<Dtype>);
    //  b_data.label_.reset(new Blob<Dtype>);
  //  LOG(INFO)<<"init queuePair_batch... data reshape dim =" <<patch_shape.size();
    //b_data.data_->Reshape(data_patch_shape);

    //b_data->data_->Reshape(data_patch_shape);
      //LOG(INFO)<<"init queuePair_batch... label reshape dim =" <<patch_shape.size();
    //if(output_label){
      //b_data.label_->Reshape(label_patch_shape);
    //  b_data->label_->Reshape(label_patch_shape);
    //}
    //  free_.push(b_data);
  }
}

template <typename Dtype>
QueuePair_Batch<Dtype>::~QueuePair_Batch() {
  Batch_data<Dtype>* bd;
  while (free_.try_pop(&bd)) {
    delete bd;
  }
  while (full_.try_pop(&bd)) {
    delete bd;
  }
}

//
template <typename Dtype>
Runner<Dtype>::Runner(const LayerParameter& param, PatchSampler<Dtype>& p_sampler)
    : param_(param),
      p_sampler_(p_sampler),
      new_queue_pairs_() {

        StartInternalThread();
}


template <typename Dtype>
Runner<Dtype>::~Runner() {
  StopInternalThread();
}

template <typename Dtype>
void Runner<Dtype>::InternalThreadEntry() {

      vector<shared_ptr<QueuePair_Batch<Dtype> > > qps;
      try {
          int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;
          if(param_.phase() == TRAIN) LOG(INFO)<<"phase =train and solover_count =" << Caffe::solver_count();
          // To ensure deterministic runs, only start running once all solvers
          // are ready. But solvers need to peek on one item during initialization,
          // so read one item, then wait for the next solver.
          LOG(INFO)<<"solver_count  = "<<solver_count << "  size of queue pairs = "<<new_queue_pairs_.size();
          //for (int i = 0; i < solver_count; ++i) {
           // LOG(INFO)<<"new_queue_pairs_.pop() = "<<i;
            shared_ptr<QueuePair_Batch<Dtype> > qp(new_queue_pairs_.pop());
            LOG(INFO)<<"size of queue  is now = " <<new_queue_pairs_.size();
            qps.push_back(qp);
			//}
          while (!must_stop()) {
            //for (int i = 0; i < solver_count; ++i) {
			 // LOG(INFO)<<"size of qps["<<i<<"] free =" <<qps[i].get()->free_.size();
			// int i=0;
              p_sampler_.ReadOnePatch(qps[0].get());
            //}
          }
        } catch (boost::thread_interrupted&) {
          // Interrupted exception is expected on shutdown
        }
   }


  template <typename Dtype>
  PatchCoordFinder<Dtype>::PatchCoordFinder(const LayerParameter& param)
  :param_(param),has_label_shape_(false),input_shape_(1,0)
  {
    CHECK(param_.patch_sampler_param().has_data_patch_shape())
    <<"patch data shape mush be given in patch_sampler_param";
    has_label_shape_=param_.patch_sampler_param().has_label_patch_shape();
    int dim_data_size = param_.patch_sampler_param().data_patch_shape().dim_size();
    if(has_label_shape_){
      int dim_label_size = param_.patch_sampler_param().label_patch_shape().dim_size();
      CHECK_EQ(dim_data_size, dim_label_size)
      <<"patch dimention axis of data and label must equal";
      }

    for (int i =0 ;i <dim_data_size;++i){
      data_shape_.push_back(param_.patch_sampler_param().data_patch_shape().dim(i));
      if(has_label_shape_){
        label_shape_.push_back(param_.patch_sampler_param().label_patch_shape().dim(i));
      }
    }

    InitRand();


    int data_patch_dims  =data_shape_.size();
    data_shape_offset_.resize(data_patch_dims,0);
    label_shape_center_.resize(data_patch_dims,0);
    if(has_label_shape_)  label_shape_offset_.resize(data_patch_dims,0);

  }

  template <typename Dtype>
   void PatchCoordFinder<Dtype>::SetInputShape(vector<int> input_shape){
          input_shape_=input_shape;
   }

   template <typename Dtype>
    vector<int> PatchCoordFinder<Dtype>:: GetDataOffset(){
     return data_shape_offset_;
   }

   template <typename Dtype>
    vector<int> PatchCoordFinder<Dtype>:: GetLabelOffset(){
     return label_shape_offset_;
   }

   template <typename Dtype>
   vector<int>  PatchCoordFinder<Dtype>::GetRandomPatchCenterCoord(){
     CHECK_GT(input_shape_[0],0)<< "input data shape has not been set, using SetInputShape member function";

    // vector<int> center_point(data_patch_dims,0);


        for(int i=0;i<data_shape_offset_.size();i++){
            int min_point     =    label_shape_[i]/2;
            int max_point     =    input_shape_[i]-min_point-1;
            //int max_point    =    input_shape_[i]-label_shape_[i]/2-1;
            int diff          =     max_point-min_point+1;
			if(diff<0){
			LOG(INFO)<<"label_shape"<<i<<"="<<label_shape_[i]<<" "<<"input_shape_[i] ="<<input_shape_[i];}
            CHECK_GT(diff,0);
            label_shape_center_[i]  =   diff==0?min_point:Rand(diff)+min_point;
           //CHECK_GE(diff,0);
            //label_shape_center_[i]  =    Rand(max_point-min_point)+min_point;
            //label_shape_center_[i]  =     Rand(diff)+min_point;

			//label_shape_offset_[i]  =     std::max(0,label_shape_center_[i] );
            //data_shape_offset_[i]   =     label_shape_center_[i] ;
            label_shape_offset_[i]  =     std::max(0,label_shape_center_[i] - min_point);
            data_shape_offset_[i]   =     label_shape_center_[i] - data_shape_[i]/2;
       }
       return label_shape_center_;
   }

  template <typename Dtype>
  void PatchCoordFinder<Dtype>::InitRand(){
     const bool needs_rand = param_.patch_sampler_param().has_data_patch_shape();
    if (needs_rand) {
      const unsigned int rng_seed = caffe_rng_rand();
      rng_.reset(new Caffe::RNG(rng_seed));
    } else {
      rng_.reset();
    }
  }

  template <typename Dtype>
  int PatchCoordFinder<Dtype>::Rand(int n) {
    CHECK(rng_);
    CHECK_GT(n, 0);
    caffe::rng_t* rng =
        static_cast<caffe::rng_t*>(rng_->generator());
    return ((*rng)() % n);
  }



  INSTANTIATE_CLASS(Runner);
  INSTANTIATE_CLASS(QueuePair_Batch);
  INSTANTIATE_CLASS(PatchSampler);
  INSTANTIATE_CLASS(PatchCoordFinder);

 }
 // namespace caffe
