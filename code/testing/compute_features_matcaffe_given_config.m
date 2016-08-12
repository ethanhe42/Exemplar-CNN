function all_features = compute_features_matcaffe_given_config(images, params)

params = setoptions(params, 'verbose', true);    
params = setoptions(params, 'do_init', true); 

if params.do_init
% init caffe network (spews logging info)
    caffe('init', params.model_def_file, params.model_file);

    % set to use GPU or CPU
    caffe('set_mode_gpu');

    % put into test mode
    caffe('set_phase_test');
end

if ~(isfield(params, 'do_not_subtract') && params.do_not_subtract)
    if isfield(params, 'mean_img')
        mean_img = params.mean_img;
    else
        if params.verbose
            fprintf('\nComputing the mean image...');
        end
        mean_img = single(mean(images,4));
    end
end

% compute the features
num_batches = ceil(size(images,4) / params.batch_size);

total_time = 0;
if params.verbose
    fprintf('\nRunning the net... ');
end
for batch = 1:num_batches
    if params.verbose
        fprintf('\nBatch %3.0d/%d', batch, num_batches);
        tic;
    end
    
    n1 = (batch-1) * params.batch_size + 1;
    n2 = min(batch * params.batch_size, size(images,4));
    curr_batch = single(images(:,:,:,n1:n2));
    if size(curr_batch, 4) < params.batch_size
        curr_batch = cat(4, curr_batch, zeros(size(curr_batch,1), size(curr_batch,2), size(curr_batch,3),...
            params.batch_size - size(curr_batch,4), 'single'));
    end
    if ~(isfield(params, 'do_not_subtract') && params.do_not_subtract)
        curr_batch = bsxfun(@minus, curr_batch, mean_img);
    end
    %curr_batch = curr_batch(:, :, [3 2 1], :);
    curr_batch = permute(curr_batch, [2 1 3 4]);
    if ~(isfield(params, 'do_not_divide') && params.do_not_divide)
        curr_batch = curr_batch/256.;
    end
    %curr_batch(1:10:end,1:10:end,:,:)
    %figure; patchShow(curr_batch, 'Color');
    input_data = {curr_batch};

    % do forward pass to get the features     
    curr_nn_maps = caffe('forward', input_data);
    
    if isfield (params, 'do_spm') && params.do_spm
        if batch == 1
            params.maps_chosen = [1:numel(curr_nn_maps)];
            for no=1:numel(curr_nn_maps)
                if numel(params.spm_pool_sizes) < no 
                    params.spm_pool_sizes{no} = [];
                elseif numel(params.spm_pool_sizes{no}) == 1 && params.spm_pool_sizes{no}(1) == 0
                    if numel(params.spm_pool_sizes_for_different_sizes) >= size(curr_nn_maps{no},1)
                        params.spm_pool_sizes{no} = params.spm_pool_sizes_for_different_sizes{size(curr_nn_maps{no},1)};
                    else
                        params.spm_pool_sizes{no} = ceil(size(curr_nn_maps{no},1)./params.spm_cells{no});
                    end
                end
            end
            if params.verbose
                for no = 1:numel(curr_nn_maps)
                    size(curr_nn_maps{no})
                    params.spm_pool_sizes{no}
                end   
            end
        end
        curr_features = make_features_from_maps_f1(curr_nn_maps, params);
        curr_features{1} = permute(curr_features{1}, [3 4 1 2]);
    else
        curr_features = curr_nn_maps;
    end
    
    if batch == 1
        for no = 1:numel(curr_features)
            if params.verbose
                no
                size(curr_features{no})
            end
            all_features{no} = zeros(size(curr_features{no},1), size(curr_features{no},2), size(curr_features{no},3), size(images,4), class(curr_features{no}));
        end
    end
    
    for no = 1:numel(curr_features)
        all_features{no}(:, :, :, n1:n2) = curr_features{no}(:, :, :, 1: n2-n1+1);
    end
    
    if params.verbose
        total_time = total_time + toc;
        fprintf('  |  elapsed time: %8.2f sec  |  estimated remaining time: %8.2f sec', total_time, total_time / batch * (num_batches - batch));
    end
    
end

if params.verbose
    fprintf('\n');
end

end
