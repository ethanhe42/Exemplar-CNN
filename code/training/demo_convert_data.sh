# you need to copy convert_binary_data.cpp to caffe tools path and compile caffe

if [ ! -f ../../data/unlabeled_training_data_DUFL_with_CNNs_STL_16000.mat ]; then
  echo "downloading the data"
  wget http://lmb.informatik.uni-freiburg.de/resources/datasets/exemplarCNN/unlabeled_training_data_DUFL_with_CNNs_STL_16000.mat -P ../../data/
fi

sh convert_data.sh ../../data/unlabeled_training_data_DUFL_with_CNNs_STL_16000.mat ../../data/STL_16000 lmdb randomize


 
