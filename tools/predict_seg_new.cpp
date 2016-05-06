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
#include <gflags/gflags.h>
#include <glog/logging.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

DEFINE_string(model,"",
    "a deployed model prototxt file");
DEFINE_string(weights, "",
    "trained model file");
DEFINE_string(data, "",
        "source input file");
DEFINE_string(predict, "", "predicted segmentation file");
DEFINE_int32(shift_axis, 0, "the patch shifting dimention along the 3d data");
DEFINE_int32(shift_num, 0,
    "total shifting times along the axis");
DEFINE_int32(shift_stride, 1,
    "shift stride along the axis");
DEFINE_int32(gpu, -1,
        "shift stride along the axis");
DEFINE_double(mean_value, 0,
                "mean value ");


class Segmentor {
 public:
  Segmentor(const string& model_file,
             const string& trained_file);

//  std::vector<Prediction> Segment(const cv::Mat& img, int N = 5);
void Segment(const string& input_hd5_file, const string& output_hd5_file);
void LoadHD5File(const char* filename);
void SaveBlob2HD5File(const char* filename, const Blob<float>& output_blob);
 private:

  void SetMean(const string& mean_file);
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
                       const string& trained_file) {

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
  LOG(INFO)<<"# channels = "<< num_channels_;
  //CHECK(num_channels_ == 3 || num_channels_ == 1)
  //  << "Input layer should have 1 or 3 channels.";
}

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

  if(d_shape.size()==3){
    //there is no num and channel in data , so appending num and channel to the data
    d_shape.insert(d_shape.begin(),1);
    d_shape.insert(d_shape.begin(),1);
    data_blob_->Reshape(d_shape);
  }else if(d_shape.size()==4){
      d_shape.insert(d_shape.begin(),1);
      data_blob_->Reshape(d_shape);
  }
  LOG(INFO) << "Successully loaded hdf5 file " << filename;

}

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

 void Segmentor::Segment(const string& input_hd5_file,const string& output_hd5_file) {
  LoadHD5File(input_hd5_file.c_str());

  TransformationNDParameter transform_param;
  //float mean_v =FLAGS_mean_value;
  transform_param.add_mean_value(FLAGS_mean_value);
  //set_mean_value(0,FLAGS_mean_value);
  DataTransformerND<float>* transformer
    = new DataTransformerND<float>(transform_param);
  Blob<float> trans_blob;
  vector<int> off_set(3,0);
  // vector<int> input_shape = data_blob_->shape();
  // vector<int > crop_shape(input_shape.begin()+2,input_shape.end());
  // Blob<float>* input_layer = net_->input_blobs()[0];
  int data_axis =2;
  vector<int> segdata_shape = data_blob_->shape();
  Blob<float>* input_blob = net_->input_blobs()[0];
  vector<int>  input_shape = input_blob->shape();
  vector<int > crop_shape(input_shape.begin()+data_axis,input_shape.end());
  int shift_data_dim_size = segdata_shape[data_axis+FLAGS_shift_axis];
  int shift_input_dim_size =crop_shape[FLAGS_shift_axis];
  LOG(INFO)<<"shift_dim size = " <<shift_input_dim_size;



  //crop_shape[FLAGS_shift_axis]=FLAGS_shift_num;
  // vector<int> inputlayer_shape =crop_shape;
  // inputlayer_shape[FLAGS_shift_axis]=FLAGS_shift_num;
  // inputlayer_shape.insert(inputlayer_shape.begin(),1);
  // inputlayer_shape.insert(inputlayer_shape.begin(),1);
  // input_layer->Reshape(inputlayer_shape);


  //input_layer->ReshapeLike(*data_blob_);

  vector<Blob<float>*> prob_blobs;
  int shif_num=0;//FLAGS_shift_num==0?shift_data_dim_size:FLAGS_shift_num;
  if (FLAGS_shift_num==0)
	  shif_num = shift_data_dim_size;
  else if(FLAGS_shift_num<0)
	  shif_num = 1;
  else
	   shif_num =FLAGS_shift_num;

//  for(int i=0;i<FLAGS_shift_num;++i){
  for(int i=0;i<shif_num;++i){
	  if(FLAGS_shift_num>=0)
		off_set[FLAGS_shift_axis]=i*FLAGS_shift_stride-shift_input_dim_size/2;
      else
		off_set[FLAGS_shift_axis]=0;
      //off_set[FLAGS_shift_axis]=i*FLAGS_shift_stride-FLAGS_shift_num/2;
      transformer->Transform(data_blob_.get(),
                              &trans_blob,
                              off_set,
                              crop_shape);
      //input_layer->CopyFrom(trans_blob);
      transformer->ApplyMean(&trans_blob,&trans_blob);
      input_blob->CopyFrom(trans_blob);
      LOG(INFO)<<"input shape "<<trans_blob.shape()[0]<<":"
      <<trans_blob.shape()[1]<<":"<<trans_blob.shape()[2]<<":"
      <<trans_blob.shape()[3]<<":"<<trans_blob.shape()[4];
      net_->Reshape();
      LOG(INFO)<<"sart forwording ...";
      net_->ForwardPrefilled();

      /* Copy the output layer to a std::vector */
      Blob<float>* output_layer = net_->output_blobs()[0];
      Blob<float>* save_blob =new Blob<float>;
      save_blob->ReshapeLike(*output_layer);
      save_blob->CopyFrom(*output_layer);
      char ext[128];
      sprintf(ext,"%s%d%s","_shift_",i,".h5");
      string out_file_name=output_hd5_file+ext;
      prob_blobs.push_back(save_blob);
      SaveBlob2HD5File(out_file_name.c_str(),*save_blob);
    //  SaveBlob2HD5File(output_hd5_file.c_str(),*output_layer);
}
  //const float* begin = output_layer->cpu_data();
  //const float* end = begin + output_layer->channels();
  //return std::vector<float>(begin, end);
}

int main(int argc, char** argv) {
  //LOG(INFO)<<"size of long ="<<sizeof(int);
  // if (argc != 8) {
  //   LOG(INFO)<<"argc ="<<argc;
  //       std::cerr << "Usage: " << argv[0]
  //             << " deploy.prototxt network.caffemodel"
  //             << " mean.binaryproto " << std::endl;
  //   return 1;
  // }

  ::google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  LOG(INFO)<<"argc ="<<argc;
  string model_file   = FLAGS_model;
  string trained_file = FLAGS_weights;
  //Classifier classifier(model_file, trained_file, mean_file, label_file);

  LOG(INFO)<<"model = : "<<model_file;
  LOG(INFO)<<"weights =  :" <<trained_file;
  const string hd5_input_file =  FLAGS_data;
  const string hd5_output_file =FLAGS_predict;
  const int gpu_id = FLAGS_gpu;
  if (gpu_id < 0) {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  } else {
    LOG(INFO) << "Using GPU " << gpu_id;
    Caffe::SetDevice(gpu_id);
    Caffe::set_mode(Caffe::GPU);
  }

  Segmentor segmentor(model_file, trained_file);

  std::cout << "---------- segmenting for "
            << hd5_input_file<< " ----------" << std::endl;
  //segmentor.loadHD5File(hd5_input_file.c_str());
  segmentor.Segment(hd5_input_file, hd5_output_file);

}
#else
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
