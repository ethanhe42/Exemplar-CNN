function [ image_names, video_num, frame_num ] = list_images(image_folders)

video_num = [];
frame_num = [];
nimage = 0;
for nfolder = 1:numel(image_folders)
    fprintf('\nFolder %d/%d: %s... ', nfolder, numel(image_folders), image_folders{nfolder});
    dd = dir(image_folders{nfolder});
    ncurrimage=0;
    for n = 1:numel(dd)
        if is_image(dd(n).name)
            ncurrimage = ncurrimage+1;
            image_names{nimage+ncurrimage} = fullfile(image_folders{nfolder}, dd(n).name);
        end
    end
    video_num = cat(1, video_num, ones(ncurrimage,1)*nfolder);
    frame_num = cat(1, frame_num, [1:ncurrimage]');
    fprintf('%d images', ncurrimage);
    nimage = nimage + ncurrimage;
end  

fprintf('\n');

end



