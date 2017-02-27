function   params   = find_Aperture ( frame ) 
% [jc,ic,r] = find_Aperture ( im )
% jc = params(1);
% ic = params(2);
% r = params(3);

% figure;imshow(frame.cdata);
% hold on;
% plot(j,i,'d');

% radius = 1;

h = imtool(frame,[0,max(max(frame))]);
 [x, y]=getpts(h); 
 close(h);
% [xo,yo,radius]=CircleFrom3Points(xk,yk);

% Call fminsearch with a random starting point.
params = fminsearch(@model, [0, 0, 1]);
% expfun accepts curve parameters as inputs, and outputs sse,
% the sum of squares error for A * exp(-lambda * xdata) - ydata, 
% and the FittedCurve. FMINSEARCH only needs sse, but we want to 
% plot the FittedCurve at the end.
    function sse = model(params)
        x0 = params(1);
        y0 = params(2);
        r  = params(3);
        
        ErrorVector = sqrt( ( x - x0 ).^2 + ( y - y0 ).^2 ) - r;
        sse = sum( ErrorVector .^ 2 );
    end
end