save_path = '../../data/training_data_STL_16000.mat';
image_list_file = '../../data/STL_image_list.mat';

%%
fprintf('\nSaving to %s?\n', save_path);
for n=1:5
    fprintf('%d... ', 6-n);
    pause(1);
end
fprintf('\n\n');

params.dataset = 'STL';
params.image_path = '/misc/lmbraid10/dosovits/Datasets/STL10/stl_unlabelled_png';


if ~exist(image_list_file, 'file')
    fprintf('\nCreating image list...');
    [image_names, video_num, frame_num] = list_images({params.image_path});
    save(image_list_file, 'image_names', 'video_num', 'frame_num');
else
    fprintf('\nLoading image list...');
    load(image_list_file, 'image_names', 'video_num', 'frame_num');
end

%%
imsize = 96;
params.subsample_probmaps = 4;
params.sample_patches_per_image = 1;
params.patchsize = 32;
params.num_patches = 16000;
params.one_patch_per_image = true;

params.nchannels = 3;
params.video_subset = unique(video_num);
params.image_subset = find(ismember(video_num, params.video_subset));

params.num_patches = min(params.num_patches, numel(params.image_subset));

params.scales = 0.8.^[3:-1:0]';
params.one_patch_per_image = true;

params.nchannels = 3;
params.video_subset = unique(video_num);
params.image_subset = find(ismember(video_num, params.video_subset));
[patches, pos] = sample_patches_multiscale(image_names(params.image_subset), params);

pos.nimg = params.image_subset(pos.nimg);
pos.video = video_num(pos.nimg);
pos.frame = frame_num(pos.nimg);
pos.cluster = [1:numel(pos.xc)]';
pos.detector = [1:numel(pos.xc)]';

%%
pos_aug3 = pos;

%%
params.num_deformations = 150;
params.scale_range = [1/sqrt(2) sqrt(2)];
params.position_range = [-0.25 0.25];
params.angle_range = [-20 20];

all_coeffs = [1 1 0.5 2 2 0.5 0.5 0.1 0.1 0.1];

pos_aug4 = augment_position_scale_color(pos_aug3,params);

curr_selection = [1:numel(pos_aug4.xc)];

pos_aug4.color1_deform = 2.^(randn(size(pos_aug4.xc))*all_coeffs(1));
pos_aug4.color2_deform = 2.^(randn(size(pos_aug4.xc))*all_coeffs(2));
pos_aug4.color3_deform = 2.^(randn(size(pos_aug4.xc))*all_coeffs(3));
pos_aug4.v_power_deform = 2.^(all_coeffs(4)*(rand(size(pos_aug4.xc))*2-1));
pos_aug4.s_power_deform = 2.^(all_coeffs(5)*(rand(size(pos_aug4.xc))*2-1));
pos_aug4.v_mult_deform = 2.^(all_coeffs(6)*(rand(size(pos_aug4.xc))*2-1));
pos_aug4.s_mult_deform = 2.^(all_coeffs(7)*(rand(size(pos_aug4.xc))*2-1));
pos_aug4.v_add_deform = all_coeffs(8)*(rand(size(pos_aug4.xc))*2-1);
pos_aug4.s_add_deform = all_coeffs(9)*(rand(size(pos_aug4.xc))*2-1);
pos_aug4.h_add_deform = all_coeffs(10)*(rand(size(pos_aug4.xc))*2-1);

[patches_aug5 pos_aug5] = get_patches_rotations(pos_aug4, params, image_names); 
%%
fprintf('\nAugmenting color...');
patches_aug5 = adjust_color(patches_aug5, pos_aug5, 10000);
patches_aug5 = power_transform(patches_aug5, pos_aug5, 10000);

images = patches_aug5;
clear patches_aug5;
rename = [1:max(pos_aug5.detector(:))];
rename(unique(pos_aug5.detector)) = [0:numel(unique(pos_aug5.detector))-1];
labels = reshape(rename(pos_aug5.detector),[],1);
fprintf('\nSaving to %s...', save_path);
save(save_path,'images','labels', '-v7.3');


