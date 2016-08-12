function [patches_out] = adjust_color(patches, pos, batchsize)

if nargin < 3
    batchsize = 10000;
end

num_batches = ceil(size(patches,4)/batchsize);

min_in = single(min(patches(:)));
max_in = single(max(patches(:)));
mean_in = single(mean(mean(mean(patches,1),2),4));

patches_out = zeros(size(patches),'uint8');

load('./color_eigenvectors.mat','v','d');
v = single(v); d = single(d);

color_deform = cat(2, pos.color1_deform, pos.color2_deform, pos.color3_deform);

tic;
for batch = 1:num_batches
    if toc > 3
        fprintf('\nBatch %d/%d', batch, num_batches); 
        tic;
    end
    n1 = (batch-1)*batchsize+1;
    n2 = min(batch*batchsize, size(patches,4));
    patches_batch = single(patches(:,:,:,n1:n2));
    patches_batch = bsxfun(@minus, patches_batch, mean_in);
    patches_batch = (patches_batch - min_in) / (max_in - min_in);
            
    curr_min = min(patches_batch(:)); curr_max = max(patches_batch(:));
    if size(patches,3) == 1
        patches_batch = repmat(patches_batch,[1 1 3 1]);
    end
    pp1 = color_transform(patches_batch, v);
    pp2 = bsxfun(@times, pp1, reshape(color_deform(n1:n2,:)',[1 1 3 size(pp1,4)]));
    pp3 = color_transform(pp2, inv(v));
    if size(patches,3) == 1
        pp3 = mean(pp3,3);
    end
    pp3(pp3 > curr_max) = curr_max;
    pp3(pp3 < curr_min) = curr_min;
    %figure; patchShow(pp3,'Color');
    pp3 = bsxfun(@plus, pp3, (mean_in - min_in) / (max_in - min_in));
    %figure; patchShow(pp3,'Color');
    pp3(pp3<0) = 0; pp3(pp3>1) = 1;
    %figure; patchShow(pp3,'Color');
    
    %{
    patches_batch = bsxfun(@power, patches_batch, reshape(pos.power_deform(n1:n2),1,1,1,[]));
    patches_batch = bsxfun(@times, patches_batch, reshape(pos.lightness_deform(n1:n2),1,1,1,[]));
    patches_out(:,:,:,n1:n2) = uint8(patches_batch * 255);
    %}
    patches_out(:,:,:,n1:n2) = uint8(pp3 * 255);
end

end





