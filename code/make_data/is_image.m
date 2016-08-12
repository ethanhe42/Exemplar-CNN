function result = is_image(filename)
    formats = {'ppm', 'PPM', 'jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG', 'tif', 'TIF', 'tiff', 'TIFF'};
    result = false;
    dots = find(filename == '.');
    if numel(dots) > 0 && dots(end) < numel(filename)
        format = filename(dots(end) + 1: end);
        if nnz(strcmp(format, formats))
            result = true;
        end
    end
end