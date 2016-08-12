function all_features = compute_features_matcaffe_new(images, params)

params.image_size = [size(images,1); size(images,2); size(images,3)];
params = make_new_config_and_net(params);

all_features = compute_features_matcaffe_given_config(images, params);

end

function params = make_new_config_and_net(params)
    % the given config file can be either ready-to use for feature
    % extraction or just a file used for training a model. Here we check
    % which is the case and create a new config and net files if necessary

    params.model_def_file = params.in_config_file;
    params.model_file = params.in_net_file;

    % read the config file.
    config_text  = fileread(params.in_config_file);
    
    % read the batch size if given
    [tokens, matches] = regexp(config_text, 'input_dim:\s*(\d*)', 'tokens', 'match');
    if numel(tokens) > 0
        % num_tokens > 0 means that this is a feature-extraction config
        batch_size = str2num(tokens{1}{1});
        image_size = [str2num(tokens{4}{1}); str2num(tokens{3}{1}); str2num(tokens{2}{1})]; 
        if isfield(params, batch_size) && params.batch_size ~= batch_size
            error('Given batch size %d does not equal the batch size in the config file %d', params.batch_size, batch_size);
        else
            params.batch_size = batch_size;
        end
        if nnz(params.image_size ~= image_size)
            error('Given image size %dx%dx%d does not equal the image size in the config file %dx%dx%d', params.image_size, image_size);
        end
    else
        % num_tokens = 0 means that this is a net-training config
        % we then turn fully connected layers to convolutional 
        
        params = fc_layers_to_conv2(params);
        params.model_def_file = params.out_config_file;
        params.model_file = params.out_net_file;
        
    end
    
    
end


