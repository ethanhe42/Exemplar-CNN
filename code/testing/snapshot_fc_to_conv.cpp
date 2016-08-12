#include <cuda_runtime.h>

#include <cstring>
#include <math.h>

#include "caffe/caffe.hpp"

using namespace caffe;

// int main(int argc, char** argv) {
//   ::google::InitGoogleLogging(argv[0]);
//   if (argc < 2) {
//     LOG(ERROR) << "Usage: snapshot_binary_to_txt solver_proto_file binary_snapshot_file";
//     return 0;
//   }
// 
//   SolverParameter solver_param;
//   ReadProtoFromTextFile(argv[1], &solver_param);
// 
//   SGDSolver<float> solver(solver_param);
//   
//   Caffe::set_mode(Caffe::Brew(solver_param.solver_mode()));
//   Caffe::set_phase(Caffe::TRAIN);
//   //solver.PreSolve();
//   solver.Snapshot_to_txt(argv[2]);
// 
//   return 0;
// }

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(ERROR) << "Usage: snapshot_fc_to_conv input_binary_snapshot_file output_binary_snapshot_file";
    return 0;
  }

  NetParameter net_param;
  LOG(INFO) << "Reading from " << argv[1];
  ReadProtoFromBinaryFile(argv[1], &net_param);
  NetParameter new_net_param(net_param);
  
  int num_layers = net_param.layers_size();
  vector<uint32_t> prev_filtersize(4,0);
  
  for (int i = 0; i < num_layers; ++i) {
    const LayerParameter& curr_layer = net_param.layers(i);
    const string& curr_layer_name = curr_layer.name();
    const LayerParameter_LayerType& curr_layer_type = curr_layer.type();
    if (curr_layer_type == LayerParameter_LayerType_CONVOLUTION || curr_layer_type == LayerParameter_LayerType_CONVOLUTION_ORTH) {
      LOG(INFO) << "Layer number " << i << ": " << curr_layer_name << ", type " << curr_layer_type << ". Has " << curr_layer.blobs_size() << " blobs.";
      Blob<float> curr_blob;      
      curr_blob.FromProto(curr_layer.blobs(0));
      prev_filtersize[0] = curr_blob.num(); 
      prev_filtersize[1] = curr_blob.channels(); 
      prev_filtersize[2] = curr_blob.height(); 
      prev_filtersize[3] = curr_blob.width();
      LOG(INFO) << "  Original blob " << 0 << ", size " << curr_blob.num() << "x" << curr_blob.channels() << "x" << curr_blob.height() << "x" << curr_blob.width();
    }
    if (curr_layer_type == LayerParameter_LayerType_INNER_PRODUCT || curr_layer_type == LayerParameter_LayerType_INNER_PRODUCT_ORTH) {
      LOG(INFO) << "Layer number " << i << ": " << curr_layer_name << ", type " << curr_layer_type << ". Has " << curr_layer.blobs_size() << " blobs.";
      LayerParameter* new_curr_layer = new_net_param.mutable_layers(i);
      //LOG(INFO) << "Prev fs: " << prev_filtersize[0] << " " << prev_filtersize[1] << " " << prev_filtersize[2] << " " << prev_filtersize[3];
      for (int j=0; j < curr_layer.blobs_size(); ++j) {
        Blob<float> curr_blob;
        vector<uint32_t> filtersize(4,0);
        curr_blob.FromProto(curr_layer.blobs(j));
        filtersize[0] = curr_blob.num(); 
        filtersize[1] = curr_blob.channels(); 
        filtersize[2] = curr_blob.height(); 
        filtersize[3] = curr_blob.width();
        LOG(INFO) << "  Original blob " << j << ", size " << curr_blob.num() << "x" << curr_blob.channels() << "x" << curr_blob.height() << "x" << curr_blob.width() << ". First element " << curr_blob.cpu_data()[0];
        if (j == 0) {
          int num_channels = prev_filtersize[0];
          int kernelsize = sqrt(filtersize[3]/num_channels);
          int num_filters = filtersize[2];
          LOG(INFO) << "Layer " << curr_layer_name << "_conv . Num channels: " << num_channels << ", kernelsize: " << kernelsize << ", num filters: " << num_filters;
          prev_filtersize[0] = num_filters;
          prev_filtersize[1] = num_channels;
          prev_filtersize[2] = kernelsize;
          prev_filtersize[3] = kernelsize;
          //LOG(INFO) << "Fs: " << filtersize[0] << " " << filtersize[1] << " " << filtersize[2] << " " << filtersize[3];
          curr_blob.Reshape_keepdata(num_filters, num_channels, kernelsize, kernelsize);
          LOG(INFO) << "  Reshaped blob " << j << ", size " << curr_blob.num() << "x" << curr_blob.channels() << "x" << curr_blob.height() << "x" << curr_blob.width() << ". First element " << curr_blob.cpu_data()[0];
          curr_blob.ToProto(new_curr_layer->mutable_blobs(j));
          new_curr_layer->set_name(curr_layer_name + "_conv");
          new_curr_layer->set_type(LayerParameter_LayerType_CONVOLUTION);
          new_curr_layer->mutable_convolution_param()->set_num_output(curr_layer.inner_product_param().num_output());
          new_curr_layer->mutable_convolution_param()->set_kernel_size(kernelsize);
          new_curr_layer->mutable_convolution_param()->set_stride(1);
          new_curr_layer->mutable_convolution_param()->set_pad((kernelsize-1)/2);
        }        
          
      }
    }
  }
  
  WriteProtoToBinaryFile(new_net_param, argv[2]);

  return 0;
}

