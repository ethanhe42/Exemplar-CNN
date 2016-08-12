function res = run_STL_folds(nn_features_all, nn_features_test_all, labels, labels_test, fold_indices, ranges1)

    res = {};
    fprintf('\n');
    for n=1:10, 
        fprintf('\rFold %d',n);
        res.fold{n} = svm_params_search({[nn_features_all(fold_indices{n},:)], [nn_features_test_all]}, {labels(fold_indices{n}),labels_test}, 'logrand_1', ranges1); 
    end

    avg_accuracy = 0;
    std_deviation = 0;
    for n=1:numel(res.fold)
        avg_accuracy = avg_accuracy+res.fold{n}.accuracy{1}(1);
    end
    avg_accuracy = avg_accuracy/numel(res.fold);
    for n=1:numel(res.fold)
        std_deviation = std_deviation + (res.fold{n}.accuracy{1}(1) - avg_accuracy)^2;
    end
    std_deviation = sqrt(std_deviation/numel(res.fold));

    res.avg_accuracy = avg_accuracy;
    res.std_deviation = std_deviation;
    fprintf('\n------ Average accuracy is %f +- %f',avg_accuracy,std_deviation);
end