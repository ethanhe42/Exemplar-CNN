// Copyright Yangqing Jia 2013
//
// This script converts the MNIST dataset to the leveldb format used
// by caffe to perform classification.
// Usage:
//    convert_mnist_data input_image_file input_label_file output_db_file
// The MNIST dataset could be downloaded at
//    http://yann.lecun.com/exdb/mnist/

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using std::string;

int IMAGE_SIZE;
int IMAGE_NBYTES;
int NUM_IMAGES;
int NUM_CHANNELS;

void read_image(std::ifstream& file, int num_bytes_toread, int& label, char* buffer) {
  uint32_t label_uint32;
  file.read(reinterpret_cast<char *>(&label_uint32), 4);
  label = static_cast<int>(label_uint32);
  file.read(buffer, num_bytes_toread);
  return;
}

void convert_dataset(const string& input_file, const string& db_folder, const string& db_backend) {

  int label;
  string value;
  uint32_t imsize[4];
  char* str_buffer;
  caffe::Datum datum;
  
  // Open new db
  // lmdb
  MDB_env *mdb_env;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;
  // leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  leveldb::WriteBatch* batch;
  
  LOG(INFO) << "Opening the db";
  
  if (db_backend == "leveldb") {  // leveldb
    LOG(INFO) << "Opening leveldb " << db_folder;
    leveldb::Status status = leveldb::DB::Open(
        options, db_folder, &db);
    CHECK(status.ok()) << "Failed to open leveldb " << db_folder;
    batch = new leveldb::WriteBatch();
  } else if (db_backend == "lmdb") {  // lmdb
    LOG(INFO) << "Opening lmdb " << db_folder;
    CHECK_EQ(mkdir(db_folder.c_str(), 0744), 0)
        << "mkdir " << db_folder << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
        << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_folder.c_str(), 0, 0664), MDB_SUCCESS)
        << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
        << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
        << "mdb_open failed";
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }  
  
    // Open files
  LOG(INFO) << "Converting " << input_file;
  std::ifstream data_file((input_file).c_str(),
      std::ios::in | std::ios::binary);
  CHECK(data_file) << "Unable to open data file " << input_file;
  
  for(int i=0; i<4; ++i)
    data_file.read(reinterpret_cast<char *>(&imsize[i]), 4);
  
  CHECK(imsize[0] == imsize[1]) << "Image size " << imsize[0] << "x" << imsize[1] << ". Images not square!";
//   CHECK(imsize[2] == 3 || imsize[2] == 1) << "Num channels " << imsize[2] << ". Not RGB or grayscale?";
  
  LOG(INFO) << "Images size " << imsize[0] << "x" << imsize[1] << "x" << imsize[2] << "x" << imsize[3];
  
  IMAGE_SIZE = imsize[0];
  NUM_CHANNELS = imsize[2];
  NUM_IMAGES = imsize[3];  
  IMAGE_NBYTES = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS;
  str_buffer = new char[IMAGE_NBYTES];
  int count = 0;
  
  datum.set_channels(NUM_CHANNELS);
  datum.set_height(IMAGE_SIZE);
  datum.set_width(IMAGE_SIZE);
  
  for (int itemid = 0; itemid < NUM_IMAGES; ++itemid) {
    read_image(data_file, IMAGE_NBYTES, label, str_buffer);
    datum.set_label(label);
    datum.set_data(str_buffer, IMAGE_NBYTES);
    datum.SerializeToString(&value);
    sprintf(str_buffer, "%08d", itemid);
    string keystr(str_buffer);
    
    // Put in db
    if (db_backend == "leveldb") {  // leveldb
      batch->Put(keystr, value);
    } else if (db_backend == "lmdb") {  // lmdb
      mdb_data.mv_size = value.size();
      mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
      mdb_key.mv_size = keystr.size();
      mdb_key.mv_data = reinterpret_cast<void*>(&keystr[0]);
      CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
          << "mdb_put failed";
    } else {
      LOG(FATAL) << "Unknown db backend " << db_backend;
    }

    if (++count % 10000 == 0 || itemid == NUM_IMAGES-1) {
      // Commit txn
      if (db_backend == "leveldb") {  // leveldb
        db->Write(leveldb::WriteOptions(), batch);
        delete batch;
        batch = new leveldb::WriteBatch();
      } else if (db_backend == "lmdb") {  // lmdb
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
            << "mdb_txn_commit failed";
        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
      } else {
        LOG(FATAL) << "Unknown db backend " << db_backend;
      }
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (db_backend == "leveldb") {  // leveldb
    delete batch;
    delete db;
  } else if (db_backend == "lmdb") {  // lmdb
    mdb_close(mdb_env, mdb_dbi);
    mdb_env_close(mdb_env);
  } else {
    LOG(FATAL) << "Unknown db backend " << db_backend;
  }
  
  delete[] str_buffer;

}

int main (int argc, char** argv) {
  if (argc != 4) {
    printf("This script converts the binary dataset created by matlab_to_binary.m to the leveldb format used\n"
           "by caffe to perform classification.\n"
           "Usage:\n"
           "    convert_matlab_data input_file output_folder db_backend\n"
           "Where the input file should have been created by matlab_to_binary.m .\n");
  } else {
    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    //cv::waitKey(0);
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]));
  }
  return 0;
}
