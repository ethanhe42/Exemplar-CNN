function [patches, pos] = sample_patches_multiscale(image_names, params, selected_images)

scales = params.scales;
subsample_probmaps = params.subsample_probmaps;
patchsize = params.patchsize;
nchannels = params.nchannels;
if ~isfield(params, 'one_patch_per_image')
    params.one_patch_per_image = false;
end

if nargin < 3
    selected_images = {};
end

sampling_probmap = {};
num_scales = numel(scales);

if mod(patchsize, subsample_probmaps) > 0
    error('subsampling_probmaps must divide patchsize');
end

if params.one_patch_per_image
        fprintf('\nOne patch per image!\n');
end

% reading in a subset of images to sample from 
if numel(selected_images) == 0
    textprogressbar('Sampling images: ');
    if params.one_patch_per_image
        num_selected_images = params.num_patches;
    else
        num_selected_images = min(16000, params.num_patches);
    end
    orig_image_num = randperm(numel(image_names), num_selected_images);
    for nimage = 1:numel(orig_image_num)
        selected_images{nimage} = imread(image_names{orig_image_num(nimage)});
        if size(selected_images{nimage},3) < 3
            selected_images{nimage} = repmat(selected_images{nimage}(:,:,1),[1 1 3]);
        end        
        textprogressbar(round(nimage/numel(orig_image_num)*100)); 
    end    
    textprogressbar('   done');
else
    fprintf('\nUsing given images to sample from\n');
    orig_image_num = [1:numel(selected_images)];
end
num_selected_images = numel(selected_images);
%figure; show_cell(selected_images); pause;

%figure; patchShow(selected_images);
%pause;

textprogressbar('Calculating probability maps: ');
image_probs = zeros(num_selected_images,1);
for nimage=1:num_selected_images
    %nimage, size(selected_images{nimage}),
    for nscale = 1:num_scales
        %nscale,
        sampling_probmap{nimage}{nscale} = imresize(single(selected_images{nimage}(1:subsample_probmaps:end,1:subsample_probmaps:end,:)),scales(nscale));
        %patchShow(sampling_probmap{nimage}{nscale});  pause;
        sampling_probmap{nimage}{nscale} = sum(sampling_probmap{nimage}{nscale},3);
        sampling_probmap{nimage}{nscale} = sampling_probmap{nimage}{nscale} - lowpassfilter(sampling_probmap{nimage}{nscale}, 2);
        %sqlp = lowpassfilter(sampling_probmap{nimage}{nscale}.^2, 3);
        %msqlp = mean(sqlp(:));
        %sampling_probmap{nimage}{nscale} = sampling_probmap{nimage}{nscale}./sqrt(sqlp + 1e0*msqlp);
        %patchShow(sampling_probmap{nimage}{nscale});  pause;
        energy_radius = min(patchsize/subsample_probmaps/4,size(sampling_probmap{nimage}{nscale},1)/4);
        sampling_probmap{nimage}{nscale} = lowpassfilter(sampling_probmap{nimage}{nscale}.^2,energy_radius);
        %sampling_probmap{nimage}{nscale} = sampling_probmap{nimage}{nscale}(1:cellSize/subsample_probmaps:end, 1:cellSize/subsample_probmaps:end, :, :);
        %patchShow(sampling_probmap{nimage}{nscale});  pause;
        borderwidth = ceil(patchsize/subsample_probmaps/2) + 1;
        sampling_probmap{nimage}{nscale}(1:borderwidth,:,:,:) = 0; sampling_probmap{nimage}{nscale}(end-borderwidth+1:end,:,:,:) = 0;
        sampling_probmap{nimage}{nscale}(:,1:borderwidth,:,:) = 0; sampling_probmap{nimage}{nscale}(:,end-borderwidth+1:end,:,:) = 0;
        sampling_probmap{nimage}{nscale}(sampling_probmap{nimage}{nscale}<0) = 0;
        %patchShow(sampling_probmap{nimage}{nscale});  pause;
        image_probs(nimage) = sum(sampling_probmap{nimage}{nscale}(:))/(size(sampling_probmap{nimage}{nscale},1)-2*borderwidth)/...
            (size(sampling_probmap{nimage}{nscale},2)-2*borderwidth)/(size(sampling_probmap{nimage}{nscale},3));
        textprogressbar(round(nimage/numel(selected_images)*100));     
    end
end
scale_probs = ones(num_scales,1);
textprogressbar('   done');
%figure; show_cell(sampling_probmap);

%% Sampling patches according to these probability maps
num_patches = params.num_patches;
patches = zeros(patchsize,patchsize,nchannels,num_patches,'uint8');

maskradius = floor(patchsize/subsample_probmaps);
%mask = fspecial('gaussian',2*[maskradius maskradius]+1, maskradius);
mask = zeros(2*maskradius+1);
%mask = max(mask(:))-mask;
%mask = mask/max(mask(:));
%figure; imshow(mask,[0 1]); pause;

textprogressbar('Sampling patches: ');
npatch = 0;
pos = struct;
while npatch<num_patches
    ncurrscale = randp(scale_probs,1);
    ncurrimage = randp(image_probs,1);
    if nnz(sampling_probmap{ncurrimage}{ncurrscale}) > 0
        currinds = randp(sampling_probmap{ncurrimage}{ncurrscale}(:),1);
        [currx, curry] = ind2sub(size(sampling_probmap{ncurrimage}{ncurrscale}),currinds);

        %[currx, curry] ./ size(sampling_probmap{ncurrimage}{ncurrscale}),
        %ncurrimage,
        currpos.nimg = orig_image_num(ncurrimage);        
        currpos.scale = ncurrscale;
        currpos.scale_value = scales(currpos.scale);        
        currpos.xc = ((currx-1)*subsample_probmaps+1) ./ currpos.scale_value;
        currpos.yc = ((curry-1)*subsample_probmaps+1) ./ currpos.scale_value;
        currpos.patchsize = patchsize ./ currpos.scale_value;        
    
        npatch = npatch+1;

        x1 = round(currpos.xc - floor(currpos.patchsize/2)); x2 = round(x1 + currpos.patchsize - 1);
        y1 = round(currpos.yc - floor(currpos.patchsize/2)); y2 = round(y1 + currpos.patchsize - 1);
        patches(:,:,:,npatch) = imresize(selected_images{ncurrimage}(x1:x2, y1:y2, :), [patchsize patchsize]);    

        for nscale=max(1,currpos.scale-3):min(num_scales,currpos.scale+3)
            coeff = scales(nscale)/scales(ncurrscale);
            x1 = round(currx*coeff) - maskradius;
            x2 = round(currx*coeff) + maskradius;
            y1 = round(curry*coeff) - maskradius;
            y2 = round(curry*coeff) + maskradius;
            x11 = max(x1,1);
            x21 = min(x2,size(sampling_probmap{ncurrimage}{nscale},1));
            y11 = max(y1,1);
            y21 = min(y2,size(sampling_probmap{ncurrimage}{nscale},2));
            %[x1,x2,y1,y2,x11,x21,y11,y21],
            if ~all(size(sampling_probmap{ncurrimage}{nscale}(x11:x21, y11:y21)) == size(mask(x11-x1+1:size(mask,1)-x2+x21,y11-y1+1:size(mask,2)-y2+y21)))
                size(sampling_probmap{ncurrimage}{nscale}(x11:x21, y11:y21)), size(mask(x11-x1+1:size(mask,1)-x2+x21,y11-y1+1:size(mask,2)-y2+y21)),
                [x1,x2,y1,y2,x11,x21,y11,y21],
            end            
            if params.one_patch_per_image
                sampling_probmap{ncurrimage}{nscale} = 0; % Don't sample from the same image twice!
            else
                sampling_probmap{ncurrimage}{nscale}(x11:x21, y11:y21) = ...
                  mask(x11-x1+1:size(mask,1)-x2+x21,y11-y1+1:size(mask,2)-y2+y21) .* sampling_probmap{ncurrimage}{nscale}(x11:x21, y11:y21);
            end
        end
        %show_cell(sampling_probmap); pause;
        textprogressbar(round(npatch/num_patches*100));
        for ff=fieldnames(currpos)'
            if ~isfield(pos,ff{1})
                pos.(ff{1}) = [];
            end
            pos.(ff{1}) = cat(1, pos.(ff{1}), currpos.(ff{1}));
        end
    end    
end
for ff=fieldnames(pos)'
     pos.(ff{1}) = pos.(ff{1})(1:num_patches);
end

textprogressbar('   done');


end


function show_cell(image_cell, finsize)
    if nargin < 2
        finsize = [64 48];
    end
    cell_size(1) = numel(image_cell);
    tmp = image_cell{1};
    while iscell(tmp)
        cell_size = [cell_size;numel(tmp)];
        tmp = tmp{1};
    end
    sp_show = zeros([finsize(1), finsize(2), size(tmp,3), prod(cell_size)],'single');
    for n=1:numel(cell_size)-1
        image_cell = [image_cell{:}];
    end
    for n=1:numel(image_cell)
        sp_show(:,:,:,n) = imresize(image_cell{n}, finsize);
    end
    sp_show = reshape(sp_show, [finsize size(sp_show,3) flipdim(cell_size,1)']);    
    patchShow(sp_show);
end

