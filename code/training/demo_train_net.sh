if [ ! -d ../../data/STL_leveldb ]; then
  echo "downloading the data"
  wget http://lmb.informatik.uni-freiburg.de/resources/datasets/exemplarCNN/unlabeled_training_data_STL_leveldb.zip -P ../../data/
fi
unzip ../../data/unlabeled_training_data_STL_leveldb.zip ../../data/

sh ./train_nn_pretrain_distr.sh 64c5-128c5-256c5-512f STL_leveldb 16000 1 16000 1200000 0.01 0.004 0.9 128 1