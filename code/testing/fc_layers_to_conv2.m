function params = fc_layers_to_conv2(params)

% params need to have fields in_net_file and in_config_file

params = setoptions(params, 'out_net_file', [params.in_net_file '_conv']);
params = setoptions(params, 'out_config_file', [params.in_config_file(1:end-9) '_conv.prototxt']);
params = setoptions(params, 'bin_config_file', [params.in_config_file(1:end-9) '.protobin']);

params = setoptions(params, 'post_pool_sizes', [1 1 1 1 1 1 1 1]);
%params = setoptions(params, 'final_featuremap_size', 8);

params = setoptions(params, 'output_blobs', {});
params = setoptions(params, 'output_blob_templates', {'conv\d*', 'fc\d*', 'fc\d*_conv'});
params = setoptions(params, 'max_num_outputs', inf);

params = setoptions(params, 'fc_to_conv_command', ['GLOG_logtostderr=1  ' fullfile(params.caffe_tools_path, 'snapshot_fc_to_conv.bin')]);
params = setoptions(params, 'proto_binary_to_text_command', ['GLOG_logtostderr=1 ' fullfile(params.caffe_tools_path, 'proto_binary_to_text.bin')]);
params = setoptions(params, 'proto_text_to_binary_command', ['GLOG_logtostderr=1 ' fullfile(params.caffe_tools_path, 'proto_text_to_binary.bin')]);
params = setoptions(params, 'fc_convolution_mode', 'valid');
params = setoptions(params, 'verbose', 'true');
params = setoptions(params, 'batch_size', 128);
params = setoptions(params, 'image_size', [32 32 3]);


%%
% convert fully connected layers of the network to corresponding
% convolutional ones

if ~exist(params.out_net_file, 'file') || ~exist(params.out_config_file, 'file')
    [status, fc_to_conv_cmdout] = system([params.fc_to_conv_command ' ' params.in_net_file ' ' params.out_net_file]);
    if params.verbose
        status
        fc_to_conv_cmdout 
    end
    if status ~= 0
        error('Smth went wrong with converting the fully connected layers to convolutional');
    end
%%

    [status, cmdout] = system([params.proto_text_to_binary_command ' ' params.in_config_file ' ' params.bin_config_file]);
    if params.verbose
        status
        cmdout 
    end

    fid = fopen(params.bin_config_file);
    buffer = fread(fid, [1 inf], '*uint8');
    fclose(fid);
    in_net_config = pb_read_caffe__NetParameter(buffer);
    out_net_config = pb_read_caffe__NetParameter();

    DATA = 5;
    CONVOLUTION = 4;
    HDF5_DATA = 9;
    INNER_PRODUCT = 14;
    POOLING = 17;
    DATA_AUGMENTATION = 36;
    DATA_LOAD_AND_AUGMENT = 37;
    INNER_PRODUCT_ORTH = 38;
    CONVOLUTION_ORTH = 41;

    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;



    new_layers = 0;
    output_blobs = {};

    for nl = 1:numel(in_net_config.layers)
        for ntop = 1:numel(in_net_config.layers(nl).top)
            curr_blob_name = in_net_config.layers(nl).top{ntop};
            if nnz(strcmp(params.output_blobs, curr_blob_name))
                if nnz(strcmp(output_blobs, curr_blob_name)) == 0
                    output_blobs = cat(1, output_blobs, curr_blob_name);
                end
            else
                for ntemp = 1:numel(params.output_blob_templates)
                    if nnz(regexp(curr_blob_name, ['^' params.output_blob_templates{ntemp} '$']))
                        if nnz(strcmp(output_blobs, curr_blob_name)) == 0
                            output_blobs = cat(1, output_blobs, curr_blob_name);
                        end
                    end
                end
            end
        end
    end

    if numel(output_blobs) > params.max_num_outputs
        output_blobs = output_blobs(1:params.max_num_outputs);
    end

    output_blobs

    max_output_layer = 0;
    for nl = 1:numel(in_net_config.layers)
        for ntop = 1:numel(in_net_config.layers(nl).top)
            curr_blob_name = in_net_config.layers(nl).top{ntop};
            if nl > max_output_layer && nnz(strcmp(curr_blob_name, output_blobs))
                max_output_layer = nl;
            end
        end
    end



    for nl = 1:max_output_layer
        curr_layer = in_net_config.layers(nl); 
        if curr_layer.type == INNER_PRODUCT || curr_layer.type == INNER_PRODUCT_ORTH
            curr_layer = pblib_set(curr_layer, 'type', CONVOLUTION);
            curr_layer = pblib_set(curr_layer, 'name', [curr_layer.name '_conv']);
            [tokens, ~] = regexp(fc_to_conv_cmdout, [curr_layer.name '.*?kernelsize:\s*(\d*)'], 'tokens', 'match');
            kernel_size = str2num(tokens{1}{1});
            conv_param = pb_read_caffe__ConvolutionParameter();
            conv_param = pblib_set(conv_param, 'num_output', curr_layer.inner_product_param.num_output);
            if strcmp(params.fc_convolution_mode, 'valid')
                conv_param = pblib_set(conv_param, 'pad', 0);
            elseif strcmp(params.fc_convolution_mode, 'same')
                conv_param = pblib_set(conv_param, 'pad', floor(kernel_size/2));
            elseif strcmp(params.fc_convolution_mode, 'full')
                conv_param = pblib_set(conv_param, 'pad', kernel_size - 1);
            end
            conv_param = pblib_set(conv_param, 'kernel_size', kernel_size);
            curr_layer = pblib_set(curr_layer, 'convolution_param', conv_param);
        end
        if ~(curr_layer.type == DATA || curr_layer.type == DATA_AUGMENTATION || curr_layer.type == DATA_LOAD_AND_AUGMENT || curr_layer.type == HDF5_DATA)
            new_layers = new_layers + 1;
            if new_layers == 1
                out_net_config = pblib_set(out_net_config, 'layers', curr_layer);
                in_data_name = curr_layer.bottom{1};
            else
                out_net_config.layers(end+1) = curr_layer;
            end
        end
    end

    out_net_config = pblib_set(out_net_config, 'input', {in_data_name});
    out_net_config = pblib_set(out_net_config, 'input_dim', [params.batch_size params.image_size(3) params.image_size(2) params.image_size(1)]);

    for nblob = 1:numel(output_blobs)
        curr_blob = output_blobs{nblob};
        curr_layer = pb_read_caffe__LayerParameter();
        curr_layer = pblib_set(curr_layer, 'name', [curr_blob '_pool']);
        curr_layer = pblib_set(curr_layer, 'type', POOLING);
        curr_layer = pblib_set(curr_layer, 'bottom', {curr_blob});
        curr_layer = pblib_set(curr_layer, 'top', {[curr_blob '_pool']});
        curr_layer = pblib_set(curr_layer, 'pooling_param', pb_read_caffe__PoolingParameter());
        curr_layer.pooling_param = pblib_set(curr_layer.pooling_param, 'pool', MAX);
        curr_layer.pooling_param = pblib_set(curr_layer.pooling_param, 'kernel_size', params.post_pool_sizes(nblob));
        curr_layer.pooling_param = pblib_set(curr_layer.pooling_param, 'stride', params.post_pool_sizes(nblob));
        out_net_config.layers(end+1) = curr_layer;
    end

    fprintf('Writing the modified config file to %s...\n', params.out_config_file);

    buffer = pblib_generic_serialize_to_string(out_net_config);
    fid = fopen(params.bin_config_file, 'w');
    fwrite(fid, buffer, 'uint8');
    fclose(fid);

    [status, cmdout] = system([params.proto_binary_to_text_command ' ' params.bin_config_file ' ' params.out_config_file]);
    if params.verbose
        status
        cmdout 
    end
else
    fprintf('\n === Reusing the existing conv network and config file ===\n');
end

    
end


    

