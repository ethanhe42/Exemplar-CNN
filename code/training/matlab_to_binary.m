function [] = matlab_to_binary(images, labels, out_file_name, randomize, append, verbose)
    if nargin < 5
        append = false;
    end
    if nargin < 6
        verbose = true;
    end

    if verbose
        fprintf('\nConverting to binary...\nWriting output to %s', out_file_name);
    end
    tic;
    if append
        fid = fopen(out_file_name, 'a');
        if verbose
            fprintf('\n Appending to existing file');
        end
    else
        fid = fopen(out_file_name, 'w');
        if verbose
            fprintf('\n Rewriting the existing file');
        end
    end
    
    if ~append
        imsize = ones(4,1);
        imsize(1:numel(size(images))) = size(images);
        fwrite(fid, uint32(imsize), 'uint32');
    end
    
    time_step = 3;
    
    if randomize == 1
        if verbose
            fprintf('\nRandomizing image order...');
        end
        image_perm = randperm(size(images,4));
        images = images(:,:,:,image_perm);
        labels = labels(image_perm);
    end
    
    images = permute(images,[2 1 3 4]);

    tic;
    for n=1:size(images,4)
        if verbose && toc > time_step
            fprintf('\nWriting image %d/%d', n, size(images,4));
            tic;
        end
        fwrite(fid, uint32(labels(n)), 'uint32');
        fwrite(fid, uint8(images(:,:,:,n)), 'uint8');    
    end
    fclose(fid);
    if verbose
        fprintf('\nTotal elapsed time: %f seconds\n', toc);
    end
    
end
