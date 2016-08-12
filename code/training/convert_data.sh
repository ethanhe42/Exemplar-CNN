# # convert data

# define variables
IN_FILE=$1
OUT_PATH=$2
DB_BACKEND=$3

RANDOMIZE=0
if [ "$4" = "randomize" ]; then
    echo "Randomly permuting the image set"
    RANDOMIZE=1
fi

TMP_BINARY=./surrogate_data.bin
DB_FOLDER=$OUT_PATH/data-$DB_BACKEND
CAFFE_TOOLS_PATH=/home/dosovits/MATLAB/toolboxes/caffe_all/.build_release/tools

echo "Input file: $IN_FILE"
echo "DB folder:  $DB_FOLDER"
echo "DB backend: $DB_BACKEND"

# .mat to binaries
echo " === Converting .mat files to binaries ==="
matlabR2013a -nodesktop -nosplash -r "\
load('"$IN_FILE"');\
matlab_to_binary(images, labels, '"$TMP_BINARY"', "$RANDOMIZE");\
exit;"  

# convert data
echo " === Converting binaries to the Caffe format === "
mkdir -p $DB_FOLDER
rm -rf $DB_FOLDER
GLOG_logtostderr=1 ${CAFFE_TOOLS_PATH}/convert_binary_data.bin \
$TMP_BINARY \
$DB_FOLDER \
$DB_BACKEND

# compute mean value
echo " === Computing mean image === "
/${CAFFE_TOOLS_PATH}/compute_image_mean.bin \
$DB_FOLDER \
$OUT_PATH/mean.binaryproto \
$DB_BACKEND

rm $TMP_BINARY


 
