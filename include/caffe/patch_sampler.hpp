#ifndef CAFFE_PATCH_SAMPLER_HPP_
#define CAFFE_PATCH_SAMPLER_HPP_

#include <map>
#include <string>
#include <vector>
#include <boost/thread.hpp>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/data_transformerND.hpp"
#include "caffe/data_provider.hpp"
#include "caffe/sample_selector.hpp"
namespace caffe {
template <typename Dtype> class PatchSampler;
 template <typename Dtype>
  class QueuePair_Batch {
   public:
    explicit QueuePair_Batch(const LayerParameter& param);
    ~QueuePair_Batch();
    BlockingQueue<Batch_data<Dtype>*> free_;
    BlockingQueue<Batch_data<Dtype>*> full_;
    DISABLE_COPY_AND_ASSIGN(QueuePair_Batch);
  };

  // A single body is created per source
  template <typename Dtype>
  class Runner : public InternalThread {
   public:
    explicit Runner(const LayerParameter& param, PatchSampler<Dtype>& p_sampler);
    //explicit Runner(const LayerParameter& param);
    virtual ~Runner();

   protected:
    void InternalThreadEntry();
    //void read_one(db::Cursor* cursor, QueuePair* qp);

    const LayerParameter param_;
    PatchSampler<Dtype>& p_sampler_;
    //const Data_provider d_provider_;
    BlockingQueue<shared_ptr<QueuePair_Batch<Dtype> > > new_queue_pairs_;

    friend class PatchSampler<Dtype>;

    //DISABLE_COPY_AND_ASSIGN(Runner);
  };

  template <typename Dtype>
  class PatchCoordFinder{
    public:
      explicit PatchCoordFinder(const LayerParameter& param);
      ~PatchCoordFinder(){};
      void SetInputShape(vector<int> input_shape);
      vector<int> GetRandomPatchCenterCoord();
      vector<int> GetDataOffset();
      vector<int> GetLabelOffset();

    protected:
      const LayerParameter param_;
      bool        has_label_shape_;
      vector<int> input_shape_;
      vector<int> label_shape_offset_;
      vector<int> data_shape_offset_;
      vector<int> label_shape_;
      vector<int> data_shape_;
      vector<int> label_shape_center_;
      shared_ptr<Caffe::RNG> rng_;
      void InitRand();
      int Rand(int n);

      //vector<int> input_shape_;
  };



/**
 * @brief warp patches from a patch_sampler to queues available to PatchSamplerLayer layers.
 * A single reading thread is created per source, even if multiple solvers
 * are running in parallel, e.g. for multi-GPU training. This makes sure
 * databases are read sequentially, and that each solver accesses a different
 * subset of the database. Data is distributed to solvers in a round-robin
 * way to keep parallel training deterministic.
 */

template <typename Dtype>
class PatchSampler {
 public:
  explicit PatchSampler(const LayerParameter& param);
  ~PatchSampler();

  inline BlockingQueue<Batch_data<Dtype>*>& free() const {
    return queue_pair_->free_;
  }
  inline BlockingQueue<Batch_data<Dtype>*>& full() const {
    return queue_pair_->full_;
  }
  void ReadOnePatch(QueuePair_Batch<Dtype>* qb );

  vector<int> patch_data_shape();
  vector<int> patch_label_shape();
 protected:
   unsigned int PrefetchRand();
  // Queue pairs are shared between a runner and its readers
  //template <typename Dtype>



  // A source is uniquely identified by its layer name + path, in case
  // the same database is read from two different locations in the net.
  static inline string source_key(const LayerParameter& param) {
    return param.name() + ":" + param.data_provider_param().data_source();
  }


  const LayerParameter param_;
  const shared_ptr<QueuePair_Batch<Dtype> > queue_pair_;
  shared_ptr<Runner<Dtype> > runner_;
  shared_ptr<Data_provider<Dtype> > d_provider_;
  shared_ptr<PatchCoordFinder<Dtype> > patch_coord_finder_;
  shared_ptr<SampleSelector<Dtype> > sample_selector_;
  shared_ptr<Caffe::RNG> prefetch_rng_;
  unsigned int patch_count_;
  unsigned int patches_per_data_batch_;
  vector<int>  dest_label_shape_;
  vector<int>  dest_data_shape_;
  shared_ptr<DataTransformerND<Dtype> > data_transformer_nd;
  boost::mutex count_m_mutex_;
  vector<int>  patch_data_shape_;
  vector<int>  patch_label_shape_;
  //data_patch_shape_
  //PeekCropCenterPoint
  //typedef typename template<typename Dtype> map<const string, boost::weak_ptr<Runner<Dtype> > > Run_container;
  static map<const string, boost::weak_ptr<Runner<Dtype> > > runners_;
  //static Run_container runners_;
  DISABLE_COPY_AND_ASSIGN(PatchSampler);
};


//  template<typename Dtype>
//  map<const string, boost::weak_ptr<Runner<Dtype> > > runners_;
}  // namespace caffe

#endif  // CAFFE_PATCH_SAMPLER_HPP_
