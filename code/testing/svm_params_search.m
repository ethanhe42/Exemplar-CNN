function [res] = svm_params_search(data, labels, method, ranges)

    global res_gl;
    if nargin < 3
        method = 'grid';
    end
    if nargin < 4
        ranges = struct();
    end
    if ~isfield(ranges,'B')
        ranges.B = [1e0];
    end
    if ~isfield(ranges,'C')
        ranges.C = [1e0];
    end
    if ~isfield(ranges,'coeff')
        ranges.coeff = [1e0];
    end
    
    rng('shuffle');
    
    if strcmp(method(1:4),'grid')
        tmp = sscanf(method,'grid_%f');
        if numel(tmp) == 1
            nfolds = tmp(1);
        else
            nfolds = 2;
        end
        res.scores = zeros(numel(ranges.B), numel(ranges.C), numel(ranges.coeff));
        res.ranges = ranges;
        total_runs = numel(ranges.B)*numel(ranges.C)*numel(ranges.coeff);
        nrun = 0;
        for nB = 1:numel(ranges.B)
            for nC = 1:numel(ranges.C)
                for ncoeff = 1:numel(ranges.coeff)
                    nrun = nrun+1;
                    if nrun==1
                        fprintf('\n');
                    end
                    curr_params(1) = ranges.B(nB);
                    curr_params(2) = ranges.C(nC);
                    curr_params(3) = ranges.coeff(ncoeff);
                    res.params{nrun} = curr_params;
                    fprintf('Run %d/%d. B=%f, C=%f, coeff=%f ', nrun, total_runs,...
                        curr_params(1), curr_params(2), curr_params(3));
                    if iscell(data)
                        fprintf('\n          Training... ');
                        res.svm{nrun} = train(double(labels{1}),sparse(double(data{1})*curr_params(3)),...
                                sprintf('-e 0.1 -B %f -c %f -q', curr_params(1), curr_params(2)));
                        [res.train_predicted_label{nrun} res.train_accuracy{nrun} res.train_decision_values{nrun}] =...
                            my_predict(labels{1}, sparse(double(data{1}))*curr_params(3), res.svm{nrun});
                        fprintf('   Testing... ');
                        [res.predicted_label{nrun} res.accuracy{nrun} res.decision_values{nrun}] =...
                            my_predict(labels{2}, sparse(double(data{2}))*curr_params(3), res.svm{nrun});
                        fprintf('\n');
                    else
                        fprintf('\n          Cross-validating %d folds...', nfolds);
                        res.accuracy{nrun} = train(double(labels),sparse(double(data)*curr_params(3)),...
                                sprintf('-v %d -B %f -c %f -q -s 1', nfolds, curr_params(1), curr_params(2)));
                    end
                    res_gl = res;
                end
            end
        end
    end
    
    if ismember(1,strfind(method,'logrand'))
        tmp = sscanf(method,'logrand_%f_%f');
        total_runs = tmp(1); 
        if numel(tmp) == 2
            nfolds = tmp(2);
        end
        if numel(ranges.B) > 2 || numel(ranges.C) > 2 || numel(ranges.coeff) > 2
            error('Too many elements in range');
        end
        for ff = fields(ranges)'            
            if numel(ranges.(ff{1})) == 1                
                ranges.(ff{1}) = repmat(ranges.(ff{1}),[2 1]);
            else
                ranges.(ff{1}) = reshape(ranges.(ff{1}), [2 1]);
            end
        end
        ranges.all = [ranges.B ranges.C ranges.coeff];
        %ranges.all
        
        for nrun=1:total_runs
            curr_params = exp(rand(1,3).*(log(ranges.all(2,:))-log(ranges.all(1,:))) + log(ranges.all(1,:)));
            res.params{nrun} = curr_params;
            if nrun==1
                fprintf('\n');
            end
            fprintf('Run %d/%d. B=%f, C=%f, coeff=%f ', nrun, total_runs,...
                        curr_params(1), curr_params(2), curr_params(3));
            if iscell(data)
                fprintf('\n          Training... ');
                res.svm{nrun} = train(double(labels{1}),sparse(double(data{1})*curr_params(3)),...
                        sprintf('-B %f -c %f -q', curr_params(1), curr_params(2)));
                [res.train_predicted_label{nrun} res.train_accuracy{nrun} res.train_decision_values{nrun}] =...
                    my_predict(labels{1}, sparse(double(data{1}))*curr_params(3), res.svm{nrun});
                fprintf('   Testing... ');
                [res.predicted_label{nrun} res.accuracy{nrun} res.decision_values{nrun}] =...
                    my_predict(labels{2}, sparse(double(data{2}))*curr_params(3), res.svm{nrun});
                fprintf('\n');
            else
                fprintf('\n          Cross-validating %d folds...',nfolds);
                res.accuracy{nrun} = train(double(labels),sparse(double(data)*curr_params(3)),...
                        sprintf('-v %d -B %f -c %f -q -s 1', nfolds, curr_params(1), curr_params(2)));
            end
            res_gl = res;
        end
    end
        

    fprintf('\n');
end

