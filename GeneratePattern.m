function I = GeneratePattern(r, m, theta, waves, pattern_length, noise)
% Generates intensity pattern for droplet with difraction parameter x and
% relative refractive index m according to Mie theory. 
%
% r -- vector or scalar of droplet radii. 
%
% m -- droplet refractive indices. Can be scalar, vector, or matrix. 
% Second dimension of m 
% corresponds to waves, that is why size(m, 2) muse always be 
%   == size(waves, 1). 
%
% waves -- vector or scalar of structures describing incident waves.
% Each element of waves is a structure with following fields:
%   * wavelength -- wavelength in the same units as radii in r
%   * theta -- wave propagation angle in scattering plane. It is counted
%   from Z-axis towards incident wavevector.
%   * polarization -- 0 if the wave is linearly polarized transversely to
%   the scattering plane and 1 if the wave is linearly polarized IN the
%   scattering plane. (Probably will be changed soon...)
%
% theta -- vector of scattering angles for which 
%   the function calculates scattered intensities. Scattering angle is an 
%   angle in the iscattering plane counted from Z-axis towards the wave vector.
%
% Function returns matrix of scattered intensities for different 
% combinations of radus and refractive indicex, angles and incident
% waves. Actually, the size of output is 
% [number of combinations (r,m)] x [number of scattering angles] x [length(waves)].
%
% r - scalar, size(m, 1) > 1.
% Combinations (r, m(i, :)) are tried.
%
% r - vector, size(m, 1) == 1. 
% Combinations (r(i), m) are used.
%
% r - vector, size(m, 1) > 1. 
% In this case size(m, 1) must be == size(r, 1). Combinations 
% (r(i), m(i,:)) are used then.
%
% If pattern_length is specified, some additional rulse apply.
% If pattern_length < length(theta), uniformly distributed random offset from
% range [0..length(theta) - pattern_length] is chosen and intensities are
% calculated for theta(1 + offset : pattern_length + offset).

if nargin < 4 || nargin > 6
    error('Invalid number of arguments');
end

if ~(isvector(waves) || isscalar(waves))
    error('waves must be vector or scalar');
end

if ndims(m) > 2
    error('Dimension of m cannot be greater than 2');
end

if size(m, 2) ~= length(waves)
    error('Second dimension of m must be equal to length of waves');
end

if ~(isvector(r) || isscalar(r))
    error('r must be either vector or scalar');
end

if isvector(r) && size(m, 1) > 1 && length(r) ~= size(m, 1)
    error('If r is a vector and m is a matrix, first dimension of m must be equal to length(r).');
end

if nargin < 5
    pattern_length = length(theta);
end

if nargin < 6
    noise = 0;
end

if isscalar(r) && size(m, 1) > 1
    r = repmat(r, 1, size(m, 1));
elseif size(m, 1) == 1
    m = repmat(m, length(r), 1);
elseif size(m, 1) ~= length(r)
    error('If r is a vector and size(m, 1) ~= 1, size(m, 1) must be equal to length(r)');
end

I = zeros(length(r), pattern_length, length(waves));

% Computing max difraction parameter
x_max = max(r) * 2 * pi / min(waves.wavelength);

% Calculating angle-dependent functions
nmax = ceil(x_max + 4 * x_max.^0.33333333 + 2);
[P T]=Mie_pt_vector_mex_C(cos(theta),int32(nmax));

I = GeneratePatternPT(r, m, P, T, waves, pattern_length, noise);