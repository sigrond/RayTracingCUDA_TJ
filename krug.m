function [j,i] = krug( varargin )
% varargin{1}=x0
% varargin{2}=y0
% varargin{3}=r0
j = -varargin { 3 } : ( varargin { 3 } / 100 ) : varargin { 3 };
i = sqrt ( varargin { 3 }^2 - j.^2 );
j = [ j -j ];
i = [ i -i ];
% j = j + varargin { 1 };
% i = i + varargin { 2 };
for k = 1 : length ( j )
    if ( j( k ) + varargin { 1 } ) < 1
        j( k ) =  1;
    else
        if ( j( k ) + varargin { 1 } ) > 640
            j( k ) =  640;
        else
            j( k ) = j( k ) + varargin { 1 };
        end

    end
if ( i( k ) + varargin { 2 } ) < 1
        i( k ) =  1;
    else
        if ( i( k ) + varargin { 2 } ) > 480
            i( k ) =  480;
        else
            i( k ) = i( k ) + varargin { 2 };
        end

    end
end
end