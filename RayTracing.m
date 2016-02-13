function P = RayTrace( P2,S )
% The function "P = RayTrace( P2,S )" - conducts rays through lens system
%
% P2 - point on the surface of the first diaphragm
% S  - structure contains the parameters of the lens
% P  - coordinates of successive intersection ray with  surfaces
%
% Calculation of the position of the sphere's center
S.Cs1  = S.l1 - S.R(1) + S.g;         
S.Cs2  = S.Cs1 + S.ll + 2*S.R(2);

P1 = S.Pk;   % droplet coordinates

v = (P2 - P1)/norm(P2 - P1); % direction vector of the line
% looking for the point of intersection of the line and lenses
%
t = (S.l1 - P2(1))/v(1); 
P3 = P2 + t*v;                            % Point in the plane parallel to the flat surface of the lens

if norm([P3(end,2),P3(end,3)]) > (S.efD/2)    % verification whether  the point inside the aperture of the lens or not
    
     % perescitywajem koordinaty 
    Kp = norm(P3(1,2:3))/(S.efD/2);
    P3(2:3) = P3(2:3)/Kp;
  v = (P3 - P1)/norm(P3 - P1); % direction vector of the line
   
end

% 
% 
% if norm(P3(2:3)) > (S.efD/2)    % verification whether  the point inside the aperture of the lens or not
%      P = [P1; P2; [NaN,NaN,NaN]];
%      return    
% end
% normal vector to the surface
n =[ 1, 0, 0 ];
% Nachodim ugol mezdu normalju i vektorom padenia i stroim nowyj
% prelomlennyj wektor
v3 = findAlpha( n, v,1,S.m2 );
%------ For intensity calculation
P(8,1:3) = acosd(dot(n,v));
% Isczem pereseczenie so sferoj
rc = SphereCross( [ P3(1) - S.Cs1, P3(2), P3(3) ], v3',S.R(1) );
% Proweriajem popal li lucz w linzu
if isnan( rc ) % lucz nie peresek sferu
    P = [P1; P2; P3;[NaN,NaN,NaN]];
    return
end

% stroim normal w punktie pereseczenija
ns = rc(1,:) / norm( rc(1,:) );
v4 = findAlpha( ns, v3',2,S.m2 );
%------ For intensity calculation
P(9,1:3) = acosd(dot(ns, v3'));
P4 = [ rc(1,1) + S.Cs1, rc(1,2), rc(1,3) ];

if norm(rc(1,2:3)) > S.D/2 
   P = [P1; P2; P3; P4;[NaN,NaN,NaN]];
   return
   
end
 
% isczem pereseczenie so wtoroj sferoj
% l - rasstojanie mezdu linzami

% centr sledujusczej sfery nachoditsia na rasstojanii 2*R+l
% znaczit wektor nado pereniesti na rasstojanie 2*R+l po osi x. Tojest
% perenosim toczku s kotoroj on wychodit


rc1 = SphereCross( [P4(1)-S.Cs2,P4(2),P4(3)], v4',S.R(2) );
if isnan( rc1 ) % lucz nie peresek sferu
    P = [P1; P2; P3; P4;[NaN,NaN,NaN]];
    return
end


P5 = rc1(2,:);
P5(1) = P5(1) + S.Cs2;

if norm(rc1(2,2:3)) > S.D/2 %lucz nie popal w linzu
    P = [P1; P2; P3; P5;[NaN,NaN,NaN]];
    return
end

% stroim normal k etoj toczkie 
ns = rc1(2,:) / norm( rc1(2,:) );

v5 = findAlpha( -ns, v4',1,S.m2 );
%------ For intensity calculation
P(10,1:3) = acosd(dot(-ns, v4'));
% rc1(2,1) - toczka pereseczenija kogda sfera nachoditsia  w naczale
% koordinat
% x0 = rc1(2,1);
% x = x0+V4(1)*t;
% posledniaja toczka eto P5
X = S.l1 + 2*S.g + S.ll;
t = ( X - P5(1) ) / v5( 1 );

P6 = P5 + v5'*t;

v6 = findAlpha( n, v5',2,S.m2 );
%------ For intensity calculation
P(11,1:3) = acosd(dot(n, v5'));
t = (S.lCCD - P6(1) ) / v6(1);

P7 = P6 + v6'*t; 

% P  = [ P1; P2; P3; P4; P5; P6; P7];
P(1,:)  =  P1;
P(2,:)  =  P2;
P(3,:)  =  P3;
P(4,:)  =  P4;
P(5,:)  =  P5;
P(6,:)  =  P6;
P(7,:)  =  P7;
% 
% ======= END OF MAIN FUNCTION ======================
    function V2 = findAlpha( n, v, p,m2 )
     %  funkcyja rasscitywajet nowyj naprawljausczij vektro dla
     % priamoj
        al1 = acosd(dot(n,v));
        % refractive index of environment and lens respectively
        m1 = 1;
%         m2 = Calculate_m(25,lambda, 'BK7');
        % Snell's law
        if p == 1
            al2 = asind( m1 * sind( al1 ) / m2 );
        else
            al2 = asind( m2 * sind( al1 ) / m1 );
        end
        % rasscitywajem ugol mezdu V1 i V2
        bet = al1 - al2;
        %
        % Stroim prelomlennyj wektor
        % stroim perpendikularnyj wektor
        S = cross( v, n );
        if norm(S) == 0 
            % wektor paralelnyj k normali net neobchodimossti scitat'
            % sistemu urawnienij
            V2 = v';
        else
            A = [ v; n; S ];
            B = [ cosd( bet ); cosd( al2 ); 0 ];
            V2 = A\B;
        end
% ------------------------------------------------------------------------
function rc = SphereCross( r, V,R )
% Eta funkcyja nachodit toczki pereseczenia lucza so sferoj
%
% r  - koordinaty toczki iz kotoroj wychdit priamaja opredelennaja
% naprawliauscim wektorom V
% R - radius sfery
%
A = sum(V.^2);
B = 2*dot(r,V);
C = sum(r.^2) - R^2;
D = B^2 -4 * A * C;
if D < 0 
rc = NaN;
    return
end
t(1) = ( -B + sqrt( D ) ) / 2 / A ;
t(2) = ( -B - sqrt( D ) ) / 2 / A ;
rc(1,:) = r + V*t(1);
rc(2,:) = r + V*t(2);
