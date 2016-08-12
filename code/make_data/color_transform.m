function img_out = color_transform(img_in, M)

    img_out = squeeze(sum(bsxfun(@times, permute(img_in,[1 2 3 5 4]), permute(M,[3 4 1 2 5])),3));

end

