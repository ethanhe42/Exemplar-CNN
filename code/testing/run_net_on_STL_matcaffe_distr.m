
params_feat.batch_size = 64; 
params_feat.do_spm = true;
params_feat.fc_convolution_mode = 'valid';
data_path = '/misc/lmbraid10/dosovits/Datasets/STL10/stl10_matlab';

% 4-quad pooling
params_feat.spm_pool_sizes_for_different_sizes = {...
    [1], [1], [2], [2], [3], ...
    [3], [4], [4], [5], [5],...
    [6], [6], [7], [7], [8]};

params_feat.spm_pool_sizes = {[0], [0], [0], [0], [0], [0]};
params_feat.verbose = 1;
params_feat.spm_pool_types = {'Max', 'Max', 'Max', 'Max', 'Max', 'Max'};
params_feat.out_config_file = [params_feat.in_config_file(1:end-9) '_conv_STL.prototxt'];


net_name_parts = strsplit(params_feat.net_name, '_');
layer_codes = strsplit(net_name_parts{1}, '-');
num_layers = numel(layer_codes);

fprintf('\nNet %s -> %d layers', params_feat.net_name, num_layers);

params_feat.max_num_outputs = num_layers;

switch num_layers
    case 3
        params_feat.post_pool_sizes = [48 24 9 17]; % for 3-layer net
    case 4
        params_feat.post_pool_sizes = [48 24 12 9 17]; % for 4-layer net
    case 5
        params_feat.post_pool_sizes = [48 24 12 12 9 17]; % for 5-layer net
    case 6
        params_feat.post_pool_sizes = [48 24 12 12 12 9 17]; % for 6-layer net
    otherwise
        error('Unknown number of layers %d', num_layers);
end

params_feat

newsize = 96;

fprintf('\nLoading the training set...');
load(fullfile(data_path,'train.mat'));
labels_train = reshape(y, [], 1); 
y = single(reshape(X',96, 96, 3,[]));
if exist('newsize','var') && newsize ~=size(y,1)
    fprintf('\nResizing to %dx%d...', newsize, newsize);
    images = zeros(newsize,newsize,3,size(y,4),'single');
    for n=1:size(y,4)
        images(:,:,:,n) = imresize(y(:,:,:,n),[newsize newsize]);
    end
else
    images = y;
end
clear y;

fprintf('\nComputing features of the training set...');
spm_features_train = compute_features_matcaffe_new(images, params_feat);
spm_features_train = squeeze(spm_features_train{1})';

%figure; imshow(spm_features_train,[]); pause;

%%
fprintf('\nLoading the testing set...');
load(fullfile(data_path,'test.mat'));
labels_test = reshape(y, [], 1); 
y = single(reshape(X',96, 96, 3,[]));
if exist('newsize','var') && newsize ~=size(y,1)
    fprintf('\nResizing to %dx%d...', newsize, newsize);
    images = zeros(newsize,newsize,3,size(y,4),'single');
    for n=1:size(y,4)
        images(:,:,:,n) = imresize(y(:,:,:,n),[newsize newsize]);
    end
else
    images = y;
end
    clear y;
fprintf('\nComputing features of the testing set...');
spm_features_test = compute_features_matcaffe_new(images, params_feat);
spm_features_test = squeeze(spm_features_test{1})';
clear images;


%%

load(fullfile(data_path, 'train.mat'), 'fold_indices');

ranges1.B=[5e0]; ranges1.C=[2e-4 2e-1]; ranges1.coeff=[1e0];
fprintf('\nRunning SVMs');
val_indices = true(size(labels_train,1),1); val_indices(fold_indices{1}) = false;
res = svm_params_search({spm_features_train(fold_indices{1},:) spm_features_train(val_indices,:)},...
    {labels_train(fold_indices{1}),labels_train(val_indices)}, 'logrand_50', ranges1);
res.dataset = 'STL-10';
res.net_name = params_feat.net_name;

%target_file = ['/misc/lmbraid10/dosovits/Krizhevsky_ConvNet/Results/STL/' net_name '_SVM.mat'];
%mkdir(fileparts(target_file));
%save(target_file, 'res');

%%
max_acc = 0;
max_ind = 0;
for n=1:numel(res.accuracy)
    if res.accuracy{n}(1) > max_acc
        max_acc = res.accuracy{n}(1);
        max_ind = n;
    end
end

ranges2.B = res.params{max_ind}(1);
ranges2.C = res.params{max_ind}(2);
ranges2.coeff = res.params{max_ind}(3);

res1 = run_STL_folds(spm_features_train, spm_features_test, labels_train, labels_test, fold_indices, ranges2);

if ~exist('out_path', 'var')
    out_path = './results_log.txt';
end
try
    outfile = fopen(out_path,'a');
    fprintf(outfile, '%-80s %7.3f+-%4.2f%%, STL\n', params_feat.net_name, res1.avg_accuracy, res1.std_deviation);
    fclose(outfile);
catch
end


