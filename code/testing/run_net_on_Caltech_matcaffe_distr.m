params_feat.batch_size = 64; 
params_feat.do_spm = 1;
params_feat.fc_convolution_mode = 'valid';
data_path = '/misc/lmbraid10/dosovits/Datasets/Caltech101';
splits_path = '.';

% SPM pooling
params_feat.spm_pool_sizes_for_different_sizes = {...
    [1], [1, 2], [1, 2, 3], [1, 2, 4], [1, 2, 5], ...
    [2, 3, 6], [2, 4, 7], [2, 4, 8], [3, 5, 9], [2, 5, 10],...
    [3, 6, 11], [3, 6, 12], [5, 7, 13], [4, 7, 14], [3, 5, 15]};

params_feat.spm_pool_sizes = {[0], [0], [0], [0], [0], []};
params_feat.verbose = 1;
params_feat.spm_pool_types = {'Max', 'Max', 'Max', 'Max', 'Max', 'Max'};
params_feat.out_config_file = [params_feat.in_config_file(1:end-9) '_conv_Caltech.prototxt'];

net_name_parts = strsplit(params_feat.net_name, '_');
layer_codes = strsplit(net_name_parts{1}, '-');
num_layers = numel(layer_codes);

fprintf('\nNet %s -> %d layers', params_feat.net_name, num_layers);

params_feat.max_num_outputs = num_layers;

switch num_layers
    case 3
        params_feat.post_pool_sizes = [19 10 4 30]; % for 3-layer net
    case 4
        params_feat.post_pool_sizes = [19 10 5 4 30]; % for 4-layer net
    case 5
        params_feat.post_pool_sizes = [19 10 5 5 4 30]; % for 5-layer net
    case 6
        params_feat.post_pool_sizes = [19 10 5 5 5 4 30]; % for 6-layer net
    otherwise
        error('Unknown number of layers %d', num_layers);
end

params_feat

%%
fprintf('\nLoading the dataset...');
load(fullfile(data_path, 'caltech101_no_back.mat'));

%%
fprintf('\nComputing features...');
spm_features = compute_features_matcaffe_new(images, params_feat);
spm_features = squeeze(spm_features{1})';


%%
fprintf('\nCrossvalidating SVM parameters...');
load(fullfile(splits_path, 'caltech101_no_back_splits.mat'), 'train_splits', 'test_splits');
%[train_splits, test_splits] = make_Caltech_train_test_splits(labels);
ranges1.B=[1e0]; ranges1.C=[2e-3 1e0]; ranges1.coeff=[1e0];
fprintf('\nRunning SVMs');
val_indices = train_splits{2}(~ismember(train_splits{2}, train_splits{1}));
res = svm_params_search({spm_features(train_splits{1},:) spm_features(val_indices,:)},...
    {labels(train_splits{1})' labels(val_indices)'}, 'logrand_30', ranges1);


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

fprintf('\nFull testing with optimal SVM parameters...');
res1 = run_Caltech_folds(spm_features, labels', train_splits, test_splits, ranges2);

for nfold = 1:numel(train_splits)
    class_accuracies{nfold} = zeros(numel(unique(labels)),1);
    curr_predicted_labels = res1.fold{nfold}.predicted_label{1};
    curr_true_labels = labels(test_splits{nfold})';
    ulabels = unique(labels);
    for n = 1:numel(ulabels)
        curr_class = find(curr_true_labels == ulabels(n));
        class_accuracies{nfold}(n) = nnz(curr_predicted_labels(curr_class) == curr_true_labels(curr_class))/numel(curr_class);
    end
    avg_accuracy(nfold) = mean(class_accuracies{nfold});
    overall_accuracy(nfold) = nnz(curr_predicted_labels == curr_true_labels) / numel(curr_true_labels);
end

avg_avg_accuracy = mean(avg_accuracy);
stdeviation_avg = sqrt(mean((avg_accuracy - avg_avg_accuracy).^2));

avg_overall_accuracy = mean(overall_accuracy);
stdeviation_overall = sqrt(mean((overall_accuracy - avg_overall_accuracy).^2));


if ~exist('out_path', 'var')
    out_path = './results_log.txt';
end
try
    outfile = fopen(out_path,'a');
    fprintf(outfile, '%-80s %7.3f+-%4.2f%% %7.3f+-%4.2f%%, Caltech-101\n', params_feat.net_name, avg_avg_accuracy*100, stdeviation_avg *100,...
        avg_overall_accuracy*100, stdeviation_overall *100);
    fclose(outfile);
catch
end




