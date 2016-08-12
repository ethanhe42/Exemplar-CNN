function out = lowpassfilter( img, sigma, batchsize, type, verbose )

if nargin < 3
    batchsize = 100;
end

if nargin < 4
    type = 'gaussian';
end

if nargin < 5
    verbose = false;
end


if ndims(img)==2
    img(:,:,1)=img;
end

if size(img,1)>1 && size(img,2)>1
    imsize = size(img);
    img = reshape(img, imsize(1), imsize(2),[]);
    out = zeros(size(img), class(img));

    dx=round(2*sigma);
    tmpimg=[img(:,dx:-1:1,:) img img(:,size(img,2):-1:size(img,2)-dx+1,:)];
    tmpimg=[tmpimg(dx:-1:1,:,:); tmpimg; tmpimg(size(img,1):-1:size(img,1)-dx+1,:,:)];
    if verbose
        fprintf('\nPreparing the filter...');
    end
    if strcmp(type,'gaussian')
        lowpass=gabor_atom([sigma sigma],0,0,[0 0],[size(tmpimg,1) size(tmpimg,2)],1);
        lowpass=cast(lowpass, class(img));
        lowpass=fft2(fftshift(lowpass));
    elseif strcmp(type,'box')
        lowpass = zeros([size(tmpimg,1) size(tmpimg,2)]);
        lowpass(1:sigma,1:sigma) = 1;
        lowpass = lowpass/sum(lowpass(:));
        lowpass = circshift(lowpass,-floor([sigma sigma]/2));
        %figure; imshow(lowpass,[]); pause;
        lowpass = fft2(lowpass);
    end
        
    %lowpass=zeros(size(tmpimg,1), size(tmpimg,2));
    %lowpass(1:2,1:2)=1; lowpass(1:2,end-1:end)=1;  lowpass(end-1:end,1:2)=1; lowpass(end-1:end,end-1:end)=1;

    nbatches=ceil(size(img,3)/batchsize);
   
    m=0;
    for batch=1:nbatches
        if verbose
            fprintf('\n  Filtering batch %d/%d...', batch, nbatches);
        end
        n=m+1;
        m=min(m+batchsize,size(img,3));
        tmplowpass = real(ifft2(bsxfun(@times,fft2(tmpimg(:,:,n:m)),lowpass)));
        out(:,:,n:m)=tmplowpass(dx+1:dx+size(img,1),dx+1:dx+size(img,2),:);
    end
    if verbose
        fprintf('\n');
    end

    out = reshape(out, imsize);
else
    imsize = size(img);
    imsize
    img = reshape(img, max(imsize), []);
    size(img)
    out = zeros(size(img));

    dx=round(2*sigma);
    tmpimg=[img(dx:-1:1,:); img; img(size(img,1):-1:size(img,1)-dx+1,:)];
    x=[-(size(tmpimg,1)-1)/2:(size(tmpimg,1)-1)/2];
    gaussian = exp(-x.^2/sigma^2);
    gaussian = gaussian/sum(gaussian);
    lowpass=fft(fftshift(gaussian))';
    nbatches=ceil(size(img,2)/batchsize);

    m=0;
    out=zeros(size(img));
    for batch=1:nbatches
        n=m+1;
        m=min(m+batchsize,size(img,2));
        tmplowpass = real(ifft(bsxfun(@times,fft(tmpimg(:,n:m)),lowpass)));
        size( out(:,n:m)), size(tmplowpass(dx+1:dx+size(img,1),:))
        out(:,n:m)=tmplowpass(dx+1:dx+size(img,1),:);
    end

    out = reshape(out, imsize);

end

