#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <caffe/util/db.hpp>
#include <caffe/util/hdf5.hpp>
#include <caffe/blob.hpp>
#include <caffe/data_transformerND.hpp>
#include <caffe/proto/caffe.pb.h>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <glog/logging.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
//typedef std::pair<string, float> Prediction;

class Segmentor {
 public:
  Segmentor(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const  bool use_gpu,
             const  int  device_id );

//  std::vector<Prediction> Segment(const cv::Mat& img, int N = 5);
void   Segment(const string& input_hd5_file, const string& output_hd5_file);
void LoadHD5File(const char* filename);
void SaveBlob2HD5File(const char* filename, const Blob<float>& output_blob);
 private:

  void SetMean(const string& mean_file);

//  std::vector<float> Predict(const cv::Mat& img);

//  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  //void Preprocess(const cv::Mat& img,
  //                std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Blob<float> > data_blob_;
//  shared_ptr<Blob<float> > label_blob_;
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  TransformationNDParameter transform_param_;
  DataTransformerND<float>* transformerND_;// = new DataTransformerND<TypeParam>(transform_param);
};

Segmentor::Segmentor(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const bool use_gpu,
                       const int  device_id=0) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  if(use_gpu){
    Caffe::set_mode(Caffe::GPU);
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
  }
  else
    Caffe::set_mode(Caffe::CPU);
#endif

  /* Load the network. */
  LOG(INFO)<<"reset net ...";
  net_.reset(new Net<float>(model_file, TEST));
  LOG(INFO)<<"done net  init ...";
  net_->CopyTrainedLayersFrom(trained_file);
  LOG(INFO)<<"trained model loaded...";
  data_blob_.reset(new Blob<float>);
  //label_blob.rest
  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
//  const vector<int>& input_shape =input_layer->shape();
  //int width =input_shape[2];
  //int height =input_shape[3];
  //input_geometry_ = cv::Size(width, height);

  /* Load the binaryproto mean file. */
//  SetMean(mean_file);

  /* Load data. */
//  loadHD5File(data_file.c_str());

//  Blob<float>* output_layer = net_->output_blobs()[0];
  //CHECK_EQ(labels_.size(), output_layer->channels())
  //  << "Number of labels is different from the output layer dimension.";
}

// static bool PairCompare(const std::pair<float, int>& lhs,
//                         const std::pair<float, int>& rhs) {
//   return lhs.first > rhs.first;
// }

/* Return the indices of the top N values of vector v. */
// static std::vector<int> Argmax(const std::vector<float>& v, int N) {
//   std::vector<std::pair<float, int> > pairs;
//   for (size_t i = 0; i < v.size(); ++i)
//     pairs.push_back(std::make_pair(v[i], i));
//   std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
//
//   std::vector<int> result;
//   for (int i = 0; i < N; ++i)
//     result.push_back(pairs[i].second);
//   return result;
// }
void Segmentor::SaveBlob2HD5File(const char* file_name, const Blob<float>& output_blob){
  LOG(INFO) << "saving HDF5 file: " << file_name;
  hid_t file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT,
                       H5P_DEFAULT);
  LOG(INFO)<<"created HD5 File...";
  CHECK_GE(file_id, 0) << "Failed to open HDF5 file" << file_name;
  const string  data_name="data";
  hdf5_save_nd_dataset(file_id, data_name, output_blob);
  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file " << file_name;
  LOG(INFO)<<"HD5 File saved";
 //  file_opened_ = true;
}
void Segmentor::LoadHD5File(const char* filename){
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  vector<string> data_set_names;
  data_set_names.push_back("data");
  //data_set_names.push_back("label");
  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  LOG(INFO) << "hdf5_load_nd_dataset ...  " << data_set_names[0];
  hdf5_load_nd_dataset(file_id, data_set_names[0].c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, data_blob_.get());
   LOG(INFO) << "finished loading hdf5 file ";
  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(data_blob_->num_axes(), 1) << "Input must have at least 1 axis.";
  vector<int> d_shape =data_blob_->shape();

  for (int i=0;i<d_shape.size();++i)
     LOG(INFO)<<"loaded data shape : " <<d_shape[i];

  if(d_shape.size()<=3){
    //there is no num and channel in data , so appending num and channel to the data
    d_shape.insert(d_shape.begin(),1);
    d_shape.insert(d_shape.begin(),1);
    data_blob_->Reshape(d_shape);
  }
  LOG(INFO) << "Successully loaded hdf5 file " << filename;

}
/* Return the top N predictions. */
// std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
//   std::vector<float> output = Predict(img);
//
//   N = std::min<int>(labels_.size(), N);
//   std::vector<int> maxN = Argmax(output, N);
//   std::vector<Prediction> predictions;
//   for (int i = 0; i < N; ++i) {
//     int idx = maxN[i];
//     predictions.push_back(std::make_pair(labels_[idx], output[idx]));
//   }
//
//   return predictions;
// }

/* Load the mean file in binaryproto format. */
void Segmentor::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

// std::vector<float> Segmentor::Segment(const cv::Mat& img) {
//   Blob<float>* input_layer = net_->input_blobs()[0];
//   input_layer->ReshapeLike(data_blob_);
//   // input_layer->Reshape(1, num_channels_,
//   //                      input_geometry_.height, input_geometry_.width);
//   /* Forward dimension change to all layers. */
//   net_->Reshape();
//
//   std::vector<cv::Mat> input_channels;
//   WrapInputLayer(&input_channels);
//
//   Preprocess(img, &input_channels);
//
//   net_->ForwardPrefilled();
//
//   /* Copy the output layer to a std::vector */
//   Blob<float>* output_layer = net_->output_blobs()[0];
//   const float* begin = output_layer->cpu_data();
//   const float* end = begin + output_layer->channels();
//   return std::vector<float>(begin, end);
// }

 void Segmentor::Segment(const string& input_hd5_file,const string& output_hd5_file) {
  LoadHD5File(input_hd5_file.c_str());
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->CopyFrom(*data_blob_.get());
  //input_layer->ReshapeLike(data_blob_);
  // input_layer->Reshape(1, num_channels_,
  //                      input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  //std::vector<cv::Mat> input_channels;
  //WrapInputLayer(&input_channels);

  //Preprocess(img, &input_channels);
  //float* input_data = input_layer->mutable_cpu_data();
  LOG(INFO)<<"sart forwording ...";
  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  SaveBlob2HD5File(output_hd5_file.c_str(),*output_layer);
  //const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels();
  //return std::vector<float>(begin, end);
}


/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
// void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
//   Blob<float>* input_layer = net_->input_blobs()[0];
//
//   int width = input_layer->width();
//   int height = input_layer->height();
//   float* input_data = input_layer->mutable_cpu_data();
//   for (int i = 0; i < input_layer->channels(); ++i) {
//     cv::Mat channel(height, width, CV_32FC1, input_data);
//     input_channels->push_back(channel);
//     input_data += width * height;
//   }
// }

// void Classifier::Preprocess(const cv::Mat& img,
//                             std::vector<cv::Mat>* input_channels) {
//   /* Convert the input image to the input image format of the network. */
//   cv::Mat sample;
//   if (img.channels() == 3 && num_channels_ == 1)
//     cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
//   else if (img.channels() == 4 && num_channels_ == 1)
//     cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
//   else if (img.channels() == 4 && num_channels_ == 3)
//     cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
//   else if (img.channels() == 1 && num_channels_ == 3)
//     cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
//   else
//     sample = img;
//
//   cv::Mat sample_resized;
//   if (sample.size() != input_geometry_)
//     cv::resize(sample, sample_resized, input_geometry_);
//   else
//     sample_resized = sample;
//
//   cv::Mat sample_float;
//   if (num_channels_ == 3)
//     sample_resized.convertTo(sample_float, CV_32FC3);
//   else
//     sample_resized.convertTo(sample_float, CV_32FC1);
//
//   cv::Mat sample_normalized;
//   cv::subtract(sample_float, mean_, sample_normalized);
//
//   /* This operation will write the separate BGR planes directly to the
//    * input layer of the network because it is wrapped by the cv::Mat
//    * objects in input_channels. */
//   cv::split(sample_normalized, *input_channels);
//
//   CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
//         == net_->input_blobs()[0]->cpu_data())
//     << "Input channels are not wrapping the input layer of the network.";
// }

int main(int argc, char** argv) {
  //LOG(INFO)<<"size of long ="<<sizeof(int);
    ::google::InitGoogleLogging(argv[0]);
     //FLAGS_logtostderr = 1;
  if (argc != 8) {
    LOG(INFO)<<"argc ="<<argc;
        std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto " << std::endl;
    return 1;
  }



  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  //string label_file   = argv[4];
  //Classifier classifier(model_file, trained_file, mean_file, label_file);


  const string hd5_input_file =  argv[4];
  const string hd5_output_file = argv[5];
  const string use_device = argv[6];
  const int device_id = atoi(argv[7]);
  const bool use_gpu=use_device=="GPU"? true:false;
  Segmentor segmentor(model_file, trained_file, mean_file,use_gpu, device_id);

  std::cout << "---------- segmenting for "
            << hd5_input_file<< " ----------" << std::endl;
  //segmentor.loadHD5File(hd5_input_file.c_str());
  segmentor.Segment( hd5_input_file, hd5_output_file);
  //cv::Mat img = cv::imread(file, -1);
  //CHECK(!img.empty()) << "Unable to decode image " << file;
//  std::vector<Prediction> predictions = classifier.Classify(img);

  /* Print the top N predictions. */
  // for (size_t i = 0; i < predictions.size(); ++i) {
  //   Prediction p = predictions[i];
  //   std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
  //             << p.first << "\"" << std::endl;
  // }
}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
