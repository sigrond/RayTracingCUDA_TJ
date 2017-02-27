function I = GeneratePatternPT(r, m, pii, tau, waves, pattern_length, noise)
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
%   * polarization -- 0 if the wave is linearly polarized transversely (s) to
%   the scattering plane and 1 if the wave is linearly polarized IN (p) the
%   scattering plane. (Probably will be changed soon...)
%
% pii, tau -- matrices of pre-calculated angle-dependent functions;
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
% If pattern_length is specified, some additional rules apply.
% If pattern_length < length(theta), uniformly distributed random offset from
% range [0..length(theta) - pattern_length] is chosen and intensities are
% calculated for theta(1 + offset : pattern_length + offset).

if nargin < 5 || nargin > 7
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

if size(pii) ~= size(tau)
    error('Sizes of Pi and Tau must be equal');
end

if nargin < 6
    pattern_length = size(pii, 2);
end

if nargin < 7
    noise = 0;
end

if isscalar(r) && size(m, 1) > 1
    r = repmat(r, 1, size(m, 1));
elseif size(m, 1) == 1
    m = repmat(m, length(r), 1);
elseif size(m, 1) ~= length(r)
    error('If r is a vector and size(m, 1) ~= 1, size(m, 1) must be equal to length(r)');
end

I = zeros(length(r), pattern_length);

for i = 1 : length(r)
    
    % Computing scattering series coefficients
    if isreal(m(i))
        i
        if i>=8600
            DebugM=complex(m(i),0)
            DebugX=complex(r(i)*2 * pi / waves(1).wavelength,0)
        end
        [a b]=Mie_ab_mex_omp(complex(m(i),0),complex(r(i)*2 * pi / waves(1).wavelength,0));
    else
        [a b]=Mie_ab_mex_omp(m(i),complex(r(i)*2 * pi / waves(1).wavelength,0));
    end
    %if i==1
    %    a(i)
    %    b(i)
    %end
        
    N = min(length(a), size(pii, 1));
    n = 1 : N;
    k = (2 * n + 1) ./ (n .* (n + 1));
    a = a .* k;
    b = b .* k;
    
    % Computing offset
    offset = floor(rand() * (size(pii, 2) - pattern_length + 1));
    offset=0;
    % Computing intensity pattern
    if waves(1).polarization == 0
        
        clearPattern = abs(a * pii(1:length(a), (1:pattern_length) + offset) ...
            + b * tau(1:length(b), (1:pattern_length) + offset)).^2;
    else
        
        clearPattern = abs(a * tau(1:length(a), (1:pattern_length) + offset) ...
            + b * pii(1:length(b), (1:pattern_length) + offset)).^2;
    end
    
    % Noising the pattern.
    I(i, :) = clearPattern; %+ normrnd(0, noise,...
    %1, length(clearPattern));
end
