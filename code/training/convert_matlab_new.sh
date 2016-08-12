IN_FILE=$1
OUT_PATH=$2
DB_BACKEND=$3

RANDOMIZE=0
if [ "$4" = "randomize" ]; then
    echo "Randomly permuting the image set"
    RANDOMIZE=1
fi

CAFFE_PATH=/home/dosovits/MATLAB/toolboxes/caffe_all
TMP_BINARY=./surrogate_data.bin
DB_FOLDER=$OUT_PATH/data-$DB_BACKEND

echo "Input file: $IN_FILE"
echo "DB folder:  $DB_FOLDER"
echo "DB backend: $DB_BACKEND"

# .mat to binaries
INIT_FOLDER=`pwd`
echo " === Converting .mat files to binaries ==="
cd /home/dosovits/MATLAB
matlabR2012b -nodesktop -nosplash -r "\
addpath(pathdef);\
load('"$IN_FILE"');\
matlab_to_binary(images, labels, '"$TMP_BINARY"', "$RANDOMIZE");\
exit;"  
cd $INIT_FOLDER

# convert data
echo " === Converting binaries to the Caffe format === "
mkdir -p $DB_FOLDER
rm -rf $DB_FOLDER
GLOG_logtostderr=1 ${CAFFE_PATH}/build/tools/convert_binary_data.bin \
$TMP_BINARY \
$DB_FOLDER \
$DB_BACKEND

# compute mean value
echo " === Computing mean image === "
${CAFFE_PATH}/build/tools/compute_image_mean.bin \
$DB_FOLDER \
$OUT_PATH/mean.binaryproto \
$DB_BACKEND

rm $TMP_BINARY


 
