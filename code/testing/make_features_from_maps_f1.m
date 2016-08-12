function [nn_features_all] = make_features_from_maps_f1(nn_maps, params)

if nargin < 2
    params = struct;
end

maps_chosen = params.maps_chosen;
pool_sizes = params.spm_pool_sizes;
pool_types = params.spm_pool_types;

for l=maps_chosen
    nn_features{l} = [];
end

for l = maps_chosen
    max_val = prctile(nn_maps{l}(:), 99.99);
    for ps = pool_sizes{l}
        if ~iscell(pool_types{l})
            pool_types{l} = {pool_types{l}};
        end
        for pt = pool_types{l}
            if ps > 1
                nn_pooled = pool_wrapper(pt{1},nn_maps{l},[ps ps], [],4); % this function is from Matthew Zeiler's Deconvolutional Networks code. Could be done in Caffe.
            else
                nn_pooled = nn_maps{l};
            end
            nn_features{l} = cat(1, nn_features{l}, reshape(nn_pooled,[],size(nn_pooled,4))/max_val*ps/max(pool_sizes{l}));
        end
    end
end

nn_features_all{1} = [];

for l=maps_chosen
    nn_features_all{1} = cat(1, nn_features_all{1}, nn_features{l});
end

clear nn_features

end

    