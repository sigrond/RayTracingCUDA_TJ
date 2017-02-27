function [err, scale] = ReferenceDistance(patterns, references)
% patterns -- array of observed intensities,
% references -- array of theoretical intensities,
% 1st dimension corresponds to time for patterns and radius for references,
% and they may differ
% in both arrays:
% 2nd -- angle,
% 3rd -- number of incident wave.
% returns:
% err -- sum of squared deviation between references and patterns * scale,
% where scale is chosen such as err is minimal.
% scale -- optimal value of scale.
if size(patterns, 2) ~= size(references, 2)
    error('Second dimensions of patterns and references must be equal.');
end
% ---------- for now we switch off waves -----------------
% if size(patterns, 3) ~= size(references, 3)
%     error('Third dimensions of patterns and references must be equal.');
% end
% 
% n_waves = size(patterns, 3); % can be 1 of course
err = zeros(size(patterns, 1), size(references, 1));
scale = zeros(size(patterns, 1), size(references, 1));
for j = 1 : size(references, 1)
    scaleTEMP=zeros(size(patterns, 1),1);
    errTEMP=zeros(size(patterns, 1),1);
    r = references(j, :);
    rSquared = sum(r.^2);
    for i = 1 : size(patterns, 1)
        p = patterns(i, :);
        scaleTEMP(i) = (p * r.') / rSquared;
        errTEMP(i) = sum((p - scaleTEMP(i) * r).^2);
    end
    scale(:,j)=scaleTEMP;
    err(:,j)=errTEMP;
    
end

