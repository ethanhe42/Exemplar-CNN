ARCH_NAME=$1
DATASET_NAME=$2
INIT_NUM_CLASSES=$3
INIT_NUM_ITER=$4
FINAL_NUM_CLASSES=$5
FINAL_NUM_ITER=$6
INIT_LR=$7
WEIGHTDECAY=$8
MOMENTUM=$9
BATCHSIZE=${10}
POSTFIX=${11}

ROOT_PATH=`cd ../..; pwd`
CAFFE_TOOLS_PATH=/home/dosovits/MATLAB/toolboxes/caffe_all/.build_release/tools
EXPERIMENT_NAME=${ARCH_NAME}_${DATASET_NAME}-${INIT_NUM_CLASSES}-${FINAL_NUM_CLASSES}_${POSTFIX}
CONFIG_TEMPLATE_PATH=$ROOT_PATH/data/nets_config/${ARCH_NAME}/template
CONFIG_OUT_PATH=$ROOT_PATH/data/nets_config/${ARCH_NAME}/result
SAVE_PATH=$ROOT_PATH/results/${EXPERIMENT_NAME}
DATA_PATH=$ROOT_PATH/data/${DATASET_NAME}
LOG_FILE=$SAVE_PATH/train_log.txt
TRAIN_NET_COMMAND=${CAFFE_TOOLS_PATH}/train_net.bin

# YOU MOST PROBABLY NEED TO CHANGE THIS:
# set up the environment 
echo "Setting up the environment" >> $LOG_FILE
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/misc/software-lin/lmbsoft/boost_1_50_0-x86_64-gcc4.4.3/include/
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/misc/software-lin/lmbsoft/boost_1_50_0-x86_64-gcc4.4.3/include/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/intel/lib/intel64:~/intel/ipp/lib/intel64/:/misc/software-lin/lmbsoft/cudatoolkit-5.5.22-x86_64/lib64/:/misc/software-lin/lmbsoft/cudatoolkit-5.0.7-x86_64/cuda/lib64/:/misc/software-lin/lmbsoft/cudatoolkit-4.2.9-x86_64/cuda/lib64/:/misc/software-lin/lmbsoft/cudatoolkit-3.2.16-x86_64/cuda/lib64/:/misc/software-lin/lmbsoft/cudatoolkit-5.0.7-x86_64/cuda/include/:/misc/software-lin/lmbsoft/boost_1_50_0-x86_64-gcc4.4.3/lib/:/home/dosovits/intel/mkl/lib/intel64/:/home/dosovits/Programs/glog/lib/:/home/dosovits/Programs/libs/
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PATH=/home/dosovits/bin:$PATH
export PATH=/home/dosovits/Programs/psched/bin:$PATH
export PATH=/home/dosovits/Programs/php/usr/bin:$PATH

PRETRAIN_WEIGHTDECAY=`echo "scale=6; ${WEIGHTDECAY}*1." | bc`
PRETRAIN_LR=`echo "scale=6; ${INIT_LR}*1." | bc`
LR_1=`echo "scale=6; ${INIT_LR}*0.4" | bc`
LR_2=`echo "scale=6; ${LR_1}*0.25" | bc`
LR_3=`echo "scale=6; ${LR_2}*0.4" | bc`
LR_4=`echo "scale=6; ${LR_3}*0.25" | bc`
NUM_ITER_1=`echo "scale=0; ${FINAL_NUM_ITER}/10" | bc`
NUM_ITER_2=`echo "scale=0; ${FINAL_NUM_ITER}/10" | bc`
NUM_ITER_3=`echo "scale=0; ${FINAL_NUM_ITER}/20" | bc`
NUM_ITER_4=`echo "scale=0; ${FINAL_NUM_ITER}/20" | bc`
END_ITER_0=`expr $INIT_NUM_ITER + $FINAL_NUM_ITER`
END_ITER_1=`expr $END_ITER_0 + $NUM_ITER_1`
END_ITER_2=`expr $END_ITER_1 + $NUM_ITER_2`
END_ITER_3=`expr $END_ITER_2 + $NUM_ITER_3`
END_ITER_4=`expr $END_ITER_3 + $NUM_ITER_4`
RECOMPUTE_MEAN=`echo "${FINAL_NUM_CLASSES}*1" | bc`

SNAPSHOT_FREQUENCY_PRETRAIN=1000
SNAPSHOT_FREQUENCY=10000

export OMP_NUM_THREADS=8

if [ "${POSTFIX}" -gt 1 ]
then
  ORIG_DATA_PATH=${DATA_PATH}
  DATA_PATH=${ORIG_DATA_PATH}_postfix${POSTFIX}
fi

# preparing configuration files

mkdir -p $CONFIG_OUT_PATH

if [ -e "${CONFIG_TEMPLATE_PATH}/train_pretrain.prototxt" ]
then
  echo "Reading from train_pretrain.prototxt"
  echo "Reading from train_pretrain.prototxt" >> $LOG_FILE
  sed -e "s:@NET_NAME@:${EXPERIMENT_NAME}_pretrain_train:g" \
  -e "s:@NUM_CLASSES@:${FINAL_NUM_CLASSES}:g" \
  -e "s:@DATASET@:${DATA_PATH}/${INIT_NUM_CLASSES}:g" \
  -e "s:@BATCHSIZE@:${BATCHSIZE}:g" \
  -e "s:@MEAN_FOLDER@:${DATA_PATH}/${FINAL_NUM_CLASSES}:g" \
  -e "s:@RECOMPUTE_MEAN@:${RECOMPUTE_MEAN}:g" \
  < ${CONFIG_TEMPLATE_PATH}/train_pretrain.prototxt > ${CONFIG_OUT_PATH}/pretrain_train.prototxt
else
  sed -e "s:@NET_NAME@:${EXPERIMENT_NAME}_pretrain_train:g" \
  -e "s:@NUM_CLASSES@:${FINAL_NUM_CLASSES}:g" \
  -e "s:@DATASET@:${DATA_PATH}/${INIT_NUM_CLASSES}:g" \
  -e "s:@BATCHSIZE@:${BATCHSIZE}:g" \
  -e "s:@MEAN_FOLDER@:${DATA_PATH}/${FINAL_NUM_CLASSES}:g" \
  -e "s:@RECOMPUTE_MEAN@:${RECOMPUTE_MEAN}:g" \
  < ${CONFIG_TEMPLATE_PATH}/train.prototxt > ${CONFIG_OUT_PATH}/pretrain_train.prototxt
fi  

sed -e "s:@NET_NAME@:${EXPERIMENT_NAME}_pretrain_test:g" \
-e "s:@NUM_CLASSES@:${FINAL_NUM_CLASSES}:g" \
-e "s:@DATASET@:$DATA_PATH/$INIT_NUM_CLASSES:g" \
-e "s:@MEAN_FOLDER@:${DATA_PATH}/${FINAL_NUM_CLASSES}:g" \
-e "s:@BATCHSIZE@:${BATCHSIZE}:g" \
< ${CONFIG_TEMPLATE_PATH}/test.prototxt > ${CONFIG_OUT_PATH}/pretrain_test.prototxt

sed -e "s:@TRAIN_NET_CONFIG@:${CONFIG_OUT_PATH}/pretrain_train.prototxt:g" \
-e "s:@TEST_NET_CONFIG@:${CONFIG_OUT_PATH}/pretrain_test.prototxt:g" \
-e "s:@EXPERIMENT_NAME@:${EXPERIMENT_NAME}_pretrain:g" \
-e "s:@LR@:${PRETRAIN_LR}:g" \
-e "s:@WEIGHT_DECAY@:${PRETRAIN_WEIGHTDECAY}:g" \
-e "s:@MOMENTUM@:${MOMENTUM}:g" \
-e "s:@MAX_ITER@:${INIT_NUM_ITER}:g" \
-e "s:@SNAPSHOT_ITER@:${SNAPSHOT_FREQUENCY_PRETRAIN}:g" \
< ${CONFIG_TEMPLATE_PATH}/solver.prototxt > ${CONFIG_OUT_PATH}/pretrain_solver.prototxt

sed -e "s:@NET_NAME@:${EXPERIMENT_NAME}_train:g" \
-e "s:@NUM_CLASSES@:${FINAL_NUM_CLASSES}:g" \
-e "s:@DATASET@:$DATA_PATH/$FINAL_NUM_CLASSES:g" \
-e "s:@MEAN_FOLDER@:${DATA_PATH}/${FINAL_NUM_CLASSES}:g" \
-e "s:@BATCHSIZE@:${BATCHSIZE}:g" \
-e "s:@RECOMPUTE_MEAN@:"0":g" \
< ${CONFIG_TEMPLATE_PATH}/train.prototxt > ${CONFIG_OUT_PATH}/train.prototxt

sed -e "s:@NET_NAME@:${EXPERIMENT_NAME}_test:g" \
-e "s:@NUM_CLASSES@:${FINAL_NUM_CLASSES}:g" \
-e "s:@DATASET@:$DATA_PATH/$FINAL_NUM_CLASSES:g" \
-e "s:@MEAN_FOLDER@:${DATA_PATH}/${FINAL_NUM_CLASSES}:g" \
-e "s:@BATCHSIZE@:${BATCHSIZE}:g" \
< ${CONFIG_TEMPLATE_PATH}/test.prototxt > ${CONFIG_OUT_PATH}/test.prototxt

sed -e "s:@TRAIN_NET_CONFIG@:${CONFIG_OUT_PATH}/train.prototxt:g" \
-e "s:@TEST_NET_CONFIG@:${CONFIG_OUT_PATH}/test.prototxt:g" \
-e "s:@EXPERIMENT_NAME@:${EXPERIMENT_NAME}:g" \
-e "s:@LR@:${INIT_LR}:g" \
-e "s:@WEIGHT_DECAY@:${WEIGHTDECAY}:g" \
-e "s:@MOMENTUM@:${MOMENTUM}:g" \
-e "s:@MAX_ITER@:${END_ITER_0}:g" \
-e "s:@SNAPSHOT_ITER@:${SNAPSHOT_FREQUENCY}:g" \
< ${CONFIG_TEMPLATE_PATH}/solver.prototxt > ${CONFIG_OUT_PATH}/solver0.prototxt

sed -e "s:@TRAIN_NET_CONFIG@:${CONFIG_OUT_PATH}/train.prototxt:g" \
-e "s:@TEST_NET_CONFIG@:${CONFIG_OUT_PATH}/test.prototxt:g" \
-e "s:@EXPERIMENT_NAME@:${EXPERIMENT_NAME}:g" \
-e "s:@LR@:${LR_1}:g" \
-e "s:@WEIGHT_DECAY@:${WEIGHTDECAY}:g" \
-e "s:@MOMENTUM@:${MOMENTUM}:g" \
-e "s:@MAX_ITER@:${END_ITER_1}:g" \
-e "s:@SNAPSHOT_ITER@:$SNAPSHOT_FREQUENCY:g" \
< ${CONFIG_TEMPLATE_PATH}/solver.prototxt > ${CONFIG_OUT_PATH}/solver1.prototxt

sed -e "s:@TRAIN_NET_CONFIG@:${CONFIG_OUT_PATH}/train.prototxt:g" \
-e "s:@TEST_NET_CONFIG@:${CONFIG_OUT_PATH}/test.prototxt:g" \
-e "s:@EXPERIMENT_NAME@:${EXPERIMENT_NAME}:g" \
-e "s:@LR@:${LR_2}:g" \
-e "s:@WEIGHT_DECAY@:${WEIGHTDECAY}:g" \
-e "s:@MOMENTUM@:${MOMENTUM}:g" \
-e "s:@MAX_ITER@:${END_ITER_2}:g" \
-e "s:@SNAPSHOT_ITER@:$SNAPSHOT_FREQUENCY:g" \
< ${CONFIG_TEMPLATE_PATH}/solver.prototxt > ${CONFIG_OUT_PATH}/solver2.prototxt

sed -e "s:@TRAIN_NET_CONFIG@:${CONFIG_OUT_PATH}/train.prototxt:g" \
-e "s:@TEST_NET_CONFIG@:${CONFIG_OUT_PATH}/test.prototxt:g" \
-e "s:@EXPERIMENT_NAME@:${EXPERIMENT_NAME}:g" \
-e "s:@LR@:${LR_3}:g" \
-e "s:@WEIGHT_DECAY@:${WEIGHTDECAY}:g" \
-e "s:@MOMENTUM@:${MOMENTUM}:g" \
-e "s:@MAX_ITER@:${END_ITER_3}:g" \
-e "s:@SNAPSHOT_ITER@:$SNAPSHOT_FREQUENCY:g" \
< ${CONFIG_TEMPLATE_PATH}/solver.prototxt > ${CONFIG_OUT_PATH}/solver3.prototxt

sed -e "s:@TRAIN_NET_CONFIG@:${CONFIG_OUT_PATH}/train.prototxt:g" \
-e "s:@TEST_NET_CONFIG@:${CONFIG_OUT_PATH}/test.prototxt:g" \
-e "s:@EXPERIMENT_NAME@:${EXPERIMENT_NAME}:g" \
-e "s:@LR@:${LR_4}:g" \
-e "s:@WEIGHT_DECAY@:${WEIGHTDECAY}:g" \
-e "s:@MOMENTUM@:${MOMENTUM}:g" \
-e "s:@MAX_ITER@:${END_ITER_4}:g" \
-e "s:@SNAPSHOT_ITER@:$SNAPSHOT_FREQUENCY:g" \
< ${CONFIG_TEMPLATE_PATH}/solver.prototxt > ${CONFIG_OUT_PATH}/solver4.prototxt


# copy training data if necessary

if [ "${POSTFIX}" -gt 1 ]
then
  if [ ! -d "$DATA_PATH" ] 
  then
    echo ""
    echo "Copying the training data"
    echo "  from $ORIG_DATA_PATH"
    echo "  to $DATA_PATH"
    echo ""
    #rm -rf $DATA_PATH
    cp -r $ORIG_DATA_PATH $DATA_PATH
  else
    echo ""
    echo "Using data from $DATA_PATH"
    echo ""
  fi
fi

# remove old stuff

INIT_PATH=`pwd`
rm -r $SAVE_PATH
mkdir -p $SAVE_PATH
cd $SAVE_PATH

# write experiment parameters to log

echo "==== Experiment parameters ====" > $LOG_FILE
echo " " >> $LOG_FILE
echo "architecture: $ARCH_NAME" >> $LOG_FILE
echo "number of classes for pre-training: $INIT_NUM_CLASSES" >> $LOG_FILE
echo "number of iterations for pre-training: $INIT_NUM_ITER" >> $LOG_FILE
echo "number of classes for training: $FINAL_NUM_CLASSES" >> $LOG_FILE
echo "number of iterations for training: $FINAL_NUM_ITER" >> $LOG_FILE
echo "initial learning rate: $INIT_LR" >> $LOG_FILE
echo "weight decay: $WEIGHTDECAY" >> $LOG_FILE
echo "momentum: $MOMENTUM" >> $LOG_FILE
echo "mini-batch size: $BATCHSIZE" >> $LOG_FILE
echo " " >> $LOG_FILE
echo "learning rate in the 1st run of training: ${INIT_LR}" >> $LOG_FILE
echo "number of iterations in the 1st run of training: $FINAL_NUM_ITER" >> $LOG_FILE
echo "learning rate in the 2nd run of training: ${LR_1}" >> $LOG_FILE
echo "number of iterations in the 2nd run of training: $NUM_ITER_1" >> $LOG_FILE
echo "learning rate in the 3rd run of training: ${LR_2}" >> $LOG_FILE
echo "number of iterations in the 3rd run of training: $NUM_ITER_2" >> $LOG_FILE
echo "learning rate in the 4th run of training: ${LR_3}" >> $LOG_FILE
echo "number of iterations in the 4th run of training: $NUM_ITER_3" >> $LOG_FILE
echo "learning rate in the 5th run of training: ${LR_4}" >> $LOG_FILE
echo "number of iterations in the 5th run of training: $NUM_ITER_4" >> $LOG_FILE
echo " " >> $LOG_FILE
echo "==== Start training ====" >> $LOG_FILE
echo " " >> $LOG_FILE

# start pretraining

echo ""
echo " ====== Start pretraining ====== "
echo "$INIT_NUM_ITER iterations with learning rate ${PRETRAIN_LR}, weight decay ${PRETRAIN_WEIGHTDECAY}, momentum ${MOMENTUM}"
echo ""

GLOG_logtostderr=1 ${TRAIN_NET_COMMAND} \
${CONFIG_OUT_PATH}/pretrain_solver.prototxt 2>> $LOG_FILE

# run real training

echo ""
echo " ====== Start real training ====== "

CURR_ITER=`expr $INIT_NUM_ITER`

echo ""
echo " ====== First run ====== "
echo "$FINAL_NUM_ITER iterations with learning rate $INIT_LR, weight decay ${WEIGHTDECAY}, momentum ${MOMENTUM}"
echo ""

GLOG_logtostderr=1 ${TRAIN_NET_COMMAND} \
${CONFIG_OUT_PATH}/solver0.prototxt ${SAVE_PATH}/${EXPERIMENT_NAME}_pretrain_iter_${INIT_NUM_ITER}.solverstate 2>> $LOG_FILE

echo ""
echo " ====== Second run ====== "
echo "$NUM_ITER_1 iterations with learning rate $LR_1, weight decay ${WEIGHTDECAY}, momentum ${MOMENTUM}"
echo ""

GLOG_logtostderr=1 ${TRAIN_NET_COMMAND} \
${CONFIG_OUT_PATH}/solver1.prototxt ${SAVE_PATH}/${EXPERIMENT_NAME}_iter_${END_ITER_0}.solverstate 2>> $LOG_FILE

echo ""
echo " ====== Third run ====== "
echo "$NUM_ITER_2 iterations with learning rate $LR_2, weight decay ${WEIGHTDECAY}, momentum ${MOMENTUM}"
echo ""

GLOG_logtostderr=1 ${TRAIN_NET_COMMAND} \
${CONFIG_OUT_PATH}/solver2.prototxt ${SAVE_PATH}/${EXPERIMENT_NAME}_iter_${END_ITER_1}.solverstate 2>> $LOG_FILE

echo ""
echo " ====== Fourth run ====== "
echo "$NUM_ITER_3 iterations with learning rate $LR_3, weight decay ${WEIGHTDECAY}, momentum ${MOMENTUM}"
echo ""

GLOG_logtostderr=1 ${TRAIN_NET_COMMAND} \
${CONFIG_OUT_PATH}/solver3.prototxt ${SAVE_PATH}/${EXPERIMENT_NAME}_iter_${END_ITER_2}.solverstate 2>> $LOG_FILE

echo ""
echo " ====== Fifth run ====== "
echo "$NUM_ITER_4 iterations with learning rate $LR_4, weight decay ${WEIGHTDECAY}, momentum ${MOMENTUM}"
echo ""

GLOG_logtostderr=1 ${TRAIN_NET_COMMAND} \
${CONFIG_OUT_PATH}/solver4.prototxt ${SAVE_PATH}/${EXPERIMENT_NAME}_iter_${END_ITER_3}.solverstate 2>> $LOG_FILE

if [ "${POSTFIX}" -gt 1 ]
then
  echo ""
  echo "Deleting the training data ${DATA_PATH}"
  echo ""
  rm -rf $DATA_PATH
fi

cd $INIT_PATH