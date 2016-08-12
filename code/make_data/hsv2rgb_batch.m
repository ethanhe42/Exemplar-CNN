function [rgb_images] = hsv2rgb_batch( hsv_images )

input_size = size(hsv_images);
rgb_images = zeros(input_size,'uint8');

for n=1:prod(input_size(4:end))
    currimg = hsv_images(:,:,:,n);
    rgb_images(:,:,:,n) = uint8(255*hsv2rgb(currimg));
end

end
