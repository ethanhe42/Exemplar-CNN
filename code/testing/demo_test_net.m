% If you want to run some other network except the example,
% 1) You need to change this path to your caffe tools path, and compile the enclosed function snapshot_fc_to_conv.cpp
params_feat.caffe_tools_path = '/home/dosovits/MATLAB/toolboxes/caffe_all/.build_release/tools';
% 2) You also need matlab protobuf https://code.google.com/p/protobuf-matlab/

addpath('/home/dosovits/MATLAB/toolboxes/caffe_all/matlab/caffe/'); % matcaffe path
addpath('/home/dosovits/MATLAB/toolboxes/liblinear-1.93/matlab/'); % liblinear path

params_feat.net_name = '64c5-128c5-256c5-512f';
params_feat.in_net_file = '../../data/trained_nets/64c5-128c5-256c5-512f_new';
params_feat.in_config_file = '../../data/nets_config/64c5-128c5-256c5-512f/test.prototxt';

out_path = '../../results/results_log.txt';
mkdir(fileparts(out_path));

run_net_on_STL_matcaffe_distr;
run_net_on_CIFAR_matcaffe_distr;
% Caltech demo won't work without pooling from Matthew Zeiler's Adaptive Deconvolutional Networks
try
    run_net_on_Caltech_matcaffe_distr;
catch
    fprintf('\nCaltech demo won''t work without pooling from Matthew Zeiler''s Adaptive Deconvolutional Networks\n');
end