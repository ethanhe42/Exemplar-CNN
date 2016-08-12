function [predicted_labels, accuracy, decision_values] = my_predict(labels, data, svm)
    
    decision_values = cat(2, data, ones(size(data,1),1)*svm.bias) * svm.w';
    [~, predicted_labels] = max(decision_values, [], 2);
    predicted_labels = svm.Label(predicted_labels);
    accuracy = nnz(predicted_labels == labels) / numel(labels) * 100;
    fprintf('Accuracy = %.2f%% (%d/%d)', accuracy, nnz(predicted_labels == labels), numel(labels));
    
end

