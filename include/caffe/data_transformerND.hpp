#ifndef CAFFE_DATA_TRANSFORMERND_HPP
#define CAFFE_DATA_TRANSFORMERND_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class CropCenterInfo{
public:
  CropCenterInfo(){};
  CropCenterInfo(const CropCenterInfo &other)
 {
   nd_off = other.nd_off;
   value = other.value;
    //std::cout << "Copy constructor was called" << std::endl;
 }
  vector<int> nd_off;
  Dtype value;
};
/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformerND {
 public:
  explicit DataTransformerND(const TransformationNDParameter& param);
  virtual ~DataTransformerND() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();

  void Transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);


  void Transform(Blob<Dtype>* input_blob,
                                         Blob<Dtype>* transformed_blob,
                                         const vector<int>& off_set);
  void Transform(Blob<Dtype>* input_blob,
                  Blob<Dtype>* transformed_blob,
                  const vector<int>& off_set,
                  const vector<int>& crop_shape);
  void ApplyMean(Blob<Dtype>* input_blob,
                    Blob<Dtype>* transformed_blob);


  // void Transform(Blob<Dtype>* input_blob,
  //                  Blob<Dtype>*  transformed_blob,
  //                  const vector<int>& off_set);

  const CropCenterInfo<Dtype> PeekCropCenterPoint(Blob<Dtype>* input_blob);
  Dtype ReadOnePoint(Blob<Dtype>* input_blob, vector<int>& pointCoord);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *
   * @param datum
   *    Datum containing the data to be transformed.
   */
  //vector<int> InferBlobShape(const Datum_ND& datum);
  /**
   * @brief Infers the shape of transformed_blob will have when
   *    the transformation is applied to the data.
   *    It uses the first element to infer the shape of the blob.
   *
   * @param datum_vector
   *    A vector of Datum containing the data to be transformed.
   */
  //vector<int> InferBlobShape(const vector<Datum_ND> & datum_vector);

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  //void Transform(const Datum_ND& datum, Dtype* transformed_data);
  // Tranformation parameters
  TransformationNDParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
