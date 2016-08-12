function [patches_aug, pos_out] = get_patches_rotations(pos_in, params, image_input)

patchsize = params.patchsize;
nchannels = params.nchannels;
%imagepath = '/misc/lmbraid10/dosovits/Tracking_GPU/car%d/';
if iscell(image_input)
    if ischar(image_input{1})
        image_input_type = 'image_names';
    else
        image_input_type = 'images_cell';
    end
else
    image_input_type = 'images_array';
end

patches_aug = zeros(patchsize,patchsize, nchannels,numel(pos_in.xc),'uint8');
whos
size(patches_aug)
npatch = 0;
goodpos = true(size(pos_in.xc));
patches_npos = zeros(numel(pos_in.xc),1);
textprogressbar('Extracting patches: ');
unique_nimg = unique(pos_in.nimg);
for n = 1:numel(unique_nimg)
    nimg = unique_nimg(n);
    curr_selection = find(pos_in.nimg == nimg);
    if strcmp(image_input_type,'image_names')
        try
            currimg = imread(image_input{nimg});
        catch
            currimg = zeros(3000, 3000, 3, 'uint8');
        end
    elseif strcmp(image_input_type,'images_cell')
        currimg = image_input{nimg};
    elseif strcmp(image_input_type,'images_array')
        currimg = image_input(:, :, :, nimg);
    end
    if size(currimg,3) < 3
        currimg = repmat(currimg(:,:,1),[1 1 3]);
    end
    for npos = reshape(curr_selection,1,[]) 
        curr_angle = pos_in.angle(npos);
        ps = pos_in.patchsize(npos);
        ps_rot = pos_in.patchsize(npos)*sqrt(2)*cos(mod(curr_angle,90)*pi/180 - pi/4);
        x1 = max(round(pos_in.xc(npos) - ps_rot/2),1);
        x2 = min(round(pos_in.xc(npos) + ps_rot/2),size(currimg,1));
        y1 = max(round(pos_in.yc(npos) - ps_rot/2),1);
        y2 = min(round(pos_in.yc(npos) + ps_rot/2),size(currimg,2));
        if y2-y1 >= ps_rot-1 && x2-x1 >= ps_rot-1 && ps_rot > 0
            npatch = npatch+1;
            patch_tmp = currimg(x1:x2, y1:y2,:);
            if curr_angle ~= 0
                patch_tmp_rot = imrotate(patch_tmp, curr_angle, 'bilinear', 'crop');
            else
                patch_tmp_rot = patch_tmp;
            end
            patch_to_save = patch_tmp_rot(max(floor(ps_rot/2 - ps/2),1) : min(ceil(ps_rot/2 + ps/2),end),...
                max(floor(ps_rot/2 - ps/2),1) : min(ceil(ps_rot/2 + ps/2),end), :);
            patches_aug(:,:,:,npatch) = uint8(imresize(patch_to_save,[patchsize patchsize]));
            patches_npos(npatch) = npos;
%         else
%             size(currimg)
%             [npos pos_in.xc(npos) pos_in.yc(npos) x1 x2 y1 y2 ps ps_rot]
%             pause
        end
    end
    textprogressbar(round(n/numel(unique_nimg)*100));       
end

patches_aug = patches_aug(:,:,:,1:npatch);
patches_npos = patches_npos(1:npatch);
for fn=fieldnames(pos_in)'
    pos_out.(fn{1}) = pos_in.(fn{1})(patches_npos);
end

textprogressbar('   done');


fprintf('\n');

end

