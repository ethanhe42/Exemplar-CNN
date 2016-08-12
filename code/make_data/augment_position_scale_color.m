function [pos_out] = augment_position_scale_color(pos_in, params)
% Augments the training set (given by positions and scales in a spatial
% pyramid) by including some neighboring positions and scales

if nargin < 2
    params = struct;
end

params = setoptions(params, 'scale_range', [0.8 1.2]);
params = setoptions(params, 'position_range', [-0.2 0.2]);
params = setoptions(params, 'angle_range', [-20 20]);
params = setoptions(params, 'num_deformations', 5);

params

pos_out = struct;

%size(params.scale_range)

for fn=fieldnames(params)'
    if numel(strfind(fn{1}, '_range')) == 1 && all(size(params.(fn{1})) == [1 2])
        params.(fn{1}) = repmat(params.(fn{1}), [numel(pos_in.xc)*params.num_deformations 1]);
    end
    if numel(strfind(fn{1}, '_range')) == 1 && all(size(params.(fn{1})) == [numel(pos_in.xc) 2])
        params.(fn{1}) = reshape(repmat(permute(params.(fn{1}),[3 1 2]), [params.num_deformations 1 1]),[],2);
    end
end

%size(params.scale_range)

for fn=fieldnames(pos_in)'
    if numel(strfind(fn{1}, '_deform')) == 0
        pos_out.(fn{1}) = reshape(repmat(reshape(pos_in.(fn{1}),1,[]),[params.num_deformations 1]),[],1);
    else
        pos_out.(fn{1}) = pos_in.(fn{1});
    end
end

%pos_out

selected_deformations = {'scale', 'xc', 'yc', 'angle'};%'lightness', 'power'};

for deform = selected_deformations
    if ~isfield(pos_in, [deform{1} '_deform'])
        %[deform{1} '_deform']
        if strcmp(deform{1},'xc') || strcmp(deform{1},'yc')
            pos_out.([deform{1} '_deform']) = rand(size(pos_out.xc)) .* ...
                (params.position_range(:,2) - params.position_range(:,1)) + params.position_range(:,1);        
        elseif strcmp(deform{1},'color')
            pos_out.([deform{1} '_deform']) = rand(size(pos_out.xc)) .* ...
                (params.position_range(:,2) - params.position_range(:,1)) + params.position_range(:,1);  
        else
            pos_out.([deform{1} '_deform']) = rand(size(pos_out.xc)) .* ...
                (params.([deform{1} '_range'])(:,2) - params.([deform{1} '_range'])(:,1)) + params.([deform{1} '_range'])(:,1);
        end
    end
end

%size(pos_out.scale_value), size(pos_out.scale_deform)
%pos_out.scale_value = pos_out.scale_value .* pos_out.scale_deform;
pos_out.patchsize = pos_out.patchsize ./ pos_out.scale_deform;
pos_out.xc = pos_out.xc + pos_out.xc_deform .* pos_out.patchsize;
pos_out.yc = pos_out.yc + pos_out.yc_deform .* pos_out.patchsize;
pos_out.angle = pos_out.angle_deform;

end

