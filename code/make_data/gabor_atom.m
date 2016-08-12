function out = gabor_atom( sigma, phi, frequency, position, windowsize, normalize )
% function out = gabor_atom( sigma, phi, frequency, position, windowsize )
%
% Returns complex Gabor filter with parameters:
%
% sigma = [sigma(1) sigma(2)] - variances along main axes
% phi - angle between the direction of the 1st axis and the vertical direction
% frequency - oscillation frequency (along the 2nd axis)
% position = [position(1) position(2)] - displacement of the center of the wavelet 
% from the center of the window
% windowsize = [windowsize(1) windowsize(2)] - size of the window (i.e. size of the 
% output filter matrix)

% create coordinate grid
halfsize=(windowsize-1)/2;
[x y]=meshgrid([-halfsize(2)-position(2):halfsize(2)-position(2)],...
    [-halfsize(1)-position(1):halfsize(1)-position(1)]);

% rotated coordinate grid
x1=x*cos(phi)+y*sin(phi);
y1=-x*sin(phi)+y*cos(phi);

% the Gaussian filter
gauss=exp(-x1.^2/sigma(2)^2/2 - y1.^2/sigma(1)^2/2);

% multiply by complex exponent and normalize to have zero mean (if frequency>0)
if abs(frequency) > 1e-4
    int1=mean2(gauss);
    out=gauss.*exp(1i*x1*frequency);
    int2=mean2(out);
    out=out-int2/int1*gauss;
else
    out=gauss.*exp(1i*x1*frequency);
end

% normalize to have L1 norm 1
if nargin > 4
    out=out/sum(abs(out(:)));
end

end

