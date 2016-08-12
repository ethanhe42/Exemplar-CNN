function [out_images] = power_transform(in_images, params, batchsize)

% if nargin < 4
%     blah = false;
% end

if nargin < 3
    batchsize = 1000;
end
num_batches = ceil(size(in_images,4)/batchsize);
out_images = zeros(size(in_images), 'uint8');

tic;
for batch=1:num_batches
    if toc > 3
        fprintf('\nBatch %d/%d', batch, num_batches); 
        tic;
    end
    n1 = (batch-1)*batchsize+1;
    n2 = min(batch*batchsize, size(in_images,4));
    in_batch = single(in_images(:,:,:,n1:n2));
    in_batch = bsxfun(@minus, in_batch, min(min(min(in_batch,[],1),[],2),[],3));
    max_in_batch = max(max(max(in_batch,[],1),[],2),[],3);
    in_batch = bsxfun(@rdivide, in_batch, max(max_in_batch, 1e-3));
    hsv_batch = rgb2hsv_batch(in_batch);
    %figure; patchShow(hsv_batch(:,:,3,:));
    %figure; hist(reshape(hsv_batch(:,:,3,:),[],1));
    hsv_batch(:,:,3,:) = bsxfun(@power, hsv_batch(:,:,3,:), reshape(params.v_power_deform(n1:n2),[1 1 1 n2-n1+1]));    
    hsv_batch(:,:,3,:) = bsxfun(@times, hsv_batch(:,:,3,:), reshape(params.v_mult_deform(n1:n2),[1 1 1 n2-n1+1]));
    hsv_batch(:,:,3,:) = bsxfun(@plus, hsv_batch(:,:,3,:), reshape(params.v_add_deform(n1:n2),[1 1 1 n2-n1+1]));
    %params.v_power_deform(n1)
    %figure; hist(reshape(hsv_batch(:,:,3,:),[],1));
    hsv_batch(:,:,2,:) = bsxfun(@power, hsv_batch(:,:,2,:), reshape(params.s_power_deform(n1:n2),[1 1 1 n2-n1+1]));
    hsv_batch(:,:,2,:) = bsxfun(@times, hsv_batch(:,:,2,:), reshape(params.s_mult_deform(n1:n2),[1 1 1 n2-n1+1]));
    hsv_batch(:,:,2,:) = bsxfun(@plus, hsv_batch(:,:,2,:), reshape(params.s_add_deform(n1:n2),[1 1 1 n2-n1+1]));
    
    hsv_batch(:,:,1,:) = mod(bsxfun(@plus, hsv_batch(:,:,1,:), reshape(params.h_add_deform(n1:n2),[1 1 1 n2-n1+1])),1);
    
    %params.v_mult_deform(n1)
    %figure; hist(reshape(hsv_batch(:,:,3,:),[],1));
    
    %figure; hist(reshape(hsv_batch(:,:,3,:),[],1)); pause;
    hsv_batch(hsv_batch < 0) = 0; hsv_batch(hsv_batch > 1) = 1;
    %figure; patchShow(hsv_batch(:,:,3,:));
    out_images(:,:,:,n1:n2) = hsv2rgb_batch(hsv_batch);
end


end

