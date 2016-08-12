params_feat.batch_size = 256; 
params_feat.do_spm = true;
params_feat.fc_convolution_mode = 'valid';
data_path = '/home/dosovits/MATLAB/data/cifar-10-batches-mat';
splits_path = '.';

% 4-quad pooling
params_feat.spm_pool_sizes_for_different_sizes = {...
    [1], [1], [2], [2], [3], ...
    [3], [4], [4], [5], [5],...
    [6], [6], [7], [7], [8]};
params_feat.spm_pool_sizes = {[0], [0], [0], [0], [0], [0]};
params_feat.verbose = 1;
params_feat.spm_pool_types = {'Max', 'Max', 'Max', 'Max', 'Max', 'Max'};
params_feat.out_config_file = [params_feat.in_config_file(1:end-9) '_conv_CIFAR.prototxt'];

FULL_SET = 1
REDUCED_SET = 1

net_name_parts = strsplit(params_feat.net_name, '_');
layer_codes = strsplit(net_name_parts{1}, '-');
num_layers = numel(layer_codes);

fprintf('\nNet %s -> %d layers', params_feat.net_name, num_layers);

params_feat.max_num_outputs = num_layers;

switch num_layers
    case 3
        params_feat.post_pool_sizes = [32 16 5 9]; % for 3-layer net
    case 4
        params_feat.post_pool_sizes = [32 16 8 5 9]; % for 4-layer net
    case 5
        params_feat.post_pool_sizes = [32 16 8 8 5 9]; % for 5-layer net
    case 6
        params_feat.post_pool_sizes = [32 16 8 8 8 5 9]; % for 6-layer net
    otherwise
        error('Unknown number of layers %d', num_layers);
end

params_feat

 
%%
fprintf('\nLoading the dataset...');
load(fullfile(data_path, 'all_cifar.mat'));

%%
newsize = 64;
fprintf('\nPreprocessing the training set...');
if newsize ~= size(trainy,1)
    images = zeros(newsize,newsize,3,size(trainy,4),'single');
    for n=1:size(trainy,4)
        images(:,:,:,n) = imresize(single(trainy(:,:,:,n)),[newsize newsize]);
    end
else
    images = single(trainy);
end

%%
fprintf('\nComputing features of the training set...');
spm_features_train = compute_features_matcaffe_new(images, params_feat);
spm_features_train = squeeze(spm_features_train{1})';
    
%%
fprintf('\nPreprocessing the test set...');
if newsize ~= size(testy,1)
    images = zeros(newsize,newsize,3,size(testy,4),'single');
    for n=1:size(testy,4)
        images(:,:,:,n) = imresize(single(testy(:,:,:,n)),[newsize newsize]);
    end
else
    images = single(testy);
end

%%
fprintf('\nComputing features of the test set...');
spm_features_test = compute_features_matcaffe_new(images, params_feat);
spm_features_test = squeeze(spm_features_test{1})';

if REDUCED_SET == 1
    fprintf('\n\n=== Reduced set (400 training samples per class) ===\n\n');
    %%
    fprintf('\nCrossvalidating SVM parameters...');
    load(fullfile(splits_path, 'all_cifar_splits.mat'), 'train_splits');
    %[train_splits, test_splits] = make_Caltech_train_test_splits(labels);
    ranges1.B=[1e0]; ranges1.C=[1e-3 1e0]; ranges1.coeff=[1e0];
    val_indices = train_splits{2}(~ismember(train_splits{2}, train_splits{1}));
    trainlabels = double(trainlabels);
    testlabels = double(testlabels);
    fprintf('\nRunning SVMs');
    res_r = svm_params_search({spm_features_train(train_splits{1},:) spm_features_train(val_indices,:)},...
        {trainlabels(train_splits{1}) trainlabels(val_indices)}, 'logrand_50', ranges1);

    %%
    max_acc = 0;
    max_ind = 0;
    for n=1:numel(res_r.accuracy)
        if res_r.accuracy{n}(1) > max_acc
            max_acc = res_r.accuracy{n}(1);
            max_ind = n;
        end
    end
    ranges2.B = res_r.params{max_ind}(1);
    ranges2.C = res_r.params{max_ind}(2);
    ranges2.coeff = res_r.params{max_ind}(3);

    fprintf('\nFull testing with optimal SVM parameters...');
    res_r1 = run_STL_folds(spm_features_train,  spm_features_test, trainlabels, testlabels, train_splits, ranges2);
    if ~exist('out_path', 'var')
        out_path = './results_log.txt';
    end
    try
        outfile = fopen(out_path,'a');
        fprintf(outfile, '%-80s %7.3f+-%4.2f%%, size = %d, CIFAR reduced set\n', params_feat.net_name, res_r1.avg_accuracy, res_r1.std_deviation, newsize);
        fclose(outfile);
    catch
    end
end

if FULL_SET == 1
    fprintf('\n\n=== Full set (5000 training samples per class) ===\n\n');
    %%
    fprintf('\nCrossvalidating SVM parameters...');
    load('/home/dosovits/MATLAB/data/cifar-10-batches-mat/all_cifar_splits.mat', 'train_splits');
    %[train_splits, test_splits] = make_Caltech_train_test_splits(labels);
    ranges1.B=[1e0]; ranges1.C=[1e-3 1e0]; ranges1.coeff=[1e0];
    trainlabels = double(trainlabels);
    testlabels = double(testlabels);
    fprintf('\nRunning SVMs');
    res_f = svm_params_search(spm_features_train,trainlabels, 'logrand_30_2', ranges1);

    %%
    max_acc = 0;
    max_ind = 0;
    for n=1:numel(res_f.accuracy)
        if res_f.accuracy{n}(1) > max_acc
            max_acc = res_f.accuracy{n}(1);
            max_ind = n;
        end
    end
    ranges2.B = res_f.params{max_ind}(1);
    ranges2.C = res_f.params{max_ind}(2);
    ranges2.coeff = res_f.params{max_ind}(3);

    fprintf('\nFull testing with optimal SVM parameters...');
    res_f1 = svm_params_search({spm_features_train,  spm_features_test}, {trainlabels, testlabels}, 'logrand_1', ranges2);
    if ~exist('out_path', 'var')
        out_path = './results_log.txt';
    end
    try
        outfile = fopen(out_path,'a');
        fprintf(outfile, '%-80s %7.3f, size = %d, CIFAR full set\n', params_feat.net_name, res_f1.accuracy{1}(1), newsize);
        fclose(outfile);
    catch
    end
end






