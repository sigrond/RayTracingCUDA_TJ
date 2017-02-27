function [ distance ] = BorderDistance( X,Y,Px,Py )
%BORDERDISTANCE funkcja ma zwracaæ odleg³oœæ punktu od ramki
%   X,Y -punkty ramki
%   Px,Py -wybrany punkt
dX=(X-Px).^2;
dY=(Y-Py).^2;
Distances=(dX+dY).^(0.5);
%distance=min(Distances);
[A2, I] = sort(Distances(:));
a=(Y(I(1))-Y(I(2)))/(X(I(1))-X(I(2)));%wspó³czynnik kierunkowy prostej przechodz¹cej przez dwa najbli¿sze punkty od wybranego punktu
b=Y(I(1))-a*X(I(1));
distance = abs(a*Px-Py+b)/(((a^2)+1)^0.5);

end

