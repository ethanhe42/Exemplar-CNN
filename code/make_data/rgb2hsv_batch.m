function [hsv_images] = rgb2hsv_batch( rgb_images )

input_size = size(rgb_images);
hsv_images = zeros(input_size,'single');
min_in = single(min(rgb_images(:)));
max_in = single(max(rgb_images(:)));

for n=1:prod(input_size(4:end))
    currimg = rgb_images(:,:,:,n);
    if ~strcmp(class(currimg),'single')
        currimg = single(currimg);
    end
    currimg = (currimg-min_in) / (max_in-min_in);
    hsv_images(:,:,:,n) = rgb2hsv(currimg);
end

end

