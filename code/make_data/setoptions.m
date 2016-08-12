function outoptions = setoptions( options, varargin )
% function outoptions = setoptions( options, varargin )
%
% outoptions = setoptions( options, 'ParamName', ParamValue )
%
% Checks if the options structure has field 'ParamName' and creates this 
% field with value ParamValue if not.

outoptions=options;

for n=1:numel(varargin)/2
    if ~isfield(options,varargin{2*n-1})
        outoptions = setfield(outoptions,varargin{2*n-1},varargin{2*n});
    end
end


end

