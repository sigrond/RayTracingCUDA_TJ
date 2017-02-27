function[l_CCD]=find_L_CCD(h_CCD_max,fi_diafragmy,lambda)
%%  Funkcja pozwala wyznaczy� zwi�zek mi�dzy h_CCD_max i l_CCD;
% wyznaczanie alpha_0_max ze �rednicy diafragmy w m
alpha_0_max=atan(fi_diafragmy/(2*16.86e-3));
%%  Wprowadzanie alpha_0_max w radianach, h_CCD_max w metrach
alpha_0=alpha_0_max; %%14.36*pi/180;
%% d�ugo�� fali w mikronach [654.25 nm (dioda laserowa)]

%% wsp�czynniki dyspersji
B1=1.03961212;
B2=0.231792344;
B3=1.01046945;
C1=6.00069867e-3;
C2=2.00179144e-2;
C3=103.560653;
%%  n_0-wpsp�czynnik za�amania o�rodka, w kt�rym znajduje obiektyw
n_0=1.0;
%%  n_s-Wsp�czynnik za�amania soczewki
n_s=sqrt((B1*lambda^2)/(lambda^2-C1)+(B2*lambda^2)/(lambda^2-C2)+(B3*lambda^2)/(lambda^2-C3)+1);
%n_s=1.51509; %dla 632.8 nm
%%  R- Promie� krzywizny powierzchni �ami�cych soczewki
R=0.018705;%%m
%%  L- Grubo�� soczewki
L=0.005;%%m
%%  f_H-D�ugo�� ogniskowa soczewki grubej liczona od p�aszczyzn g��wnych:
f_H=1/((n_s-1)*(2/R-L*(n_s-1)/(n_s*R^2)));
%%f=0.01729+0.0025; %%m  od czo�a + p� grubo�ci
%% odleg��� p�aszczyzny g��wnej od czo�a soczewki
VH=L*f_H*(n_s-1)/(R*n_s);
%%  f-D�ugo�� ogniskowa soczewki grubej liczona od �rodka
f=f_H-VH+L/2;
L_TOT=0.04257;%%-Oleg�o�� mi�dzy soczewkami;%%m

xx_1=(2*f-L)/(2*R)+1;

beta_1=asin((n_0/n_s)*xx_1*sin(alpha_0));

beta_0=asin(xx_1*sin(alpha_0));

beta_00=alpha_0+beta_1-beta_0;
beta_2=beta_0-alpha_0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_0=R*sin(beta_2);
x_0=R*(1-cos(beta_2));
a_0=h_0*cot(beta_00);
a_00=a_0-x_0+L-R;
h_prim=a_00*sin(beta_00);

beta_3=asin(h_prim/R);
beta_4=asin((n_s/n_0)*sin(beta_3));

beta_11=beta_4-beta_3-beta_00;
beta_3_prim=pi/2-beta_00;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h_1=R*cos(beta_3_prim-beta_3);
x_1=R*(1-sin(beta_3_prim-beta_3));
b_1=h_1*cot(beta_11);

b_11=b_1-x_1-L_TOT-R;
h_11=b_11*sin(beta_11);

beta_5=asin(h_11/R);
beta_6=asin((n_0/n_s)*sin(beta_5));

h_22=R*sin(beta_6);
gama=beta_11+beta_5;

%%h_2=R*sin(gama);
beta_22=gama-beta_6;
b_22=h_22/sin(beta_22);
b_22_prim=b_22+2*R-L;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h_2_prim=b_22_prim*sin(beta_22);
beta_7=asin(h_2_prim/R);
beta_8=asin((n_s/n_0)*sin(beta_7));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_9=beta_8-beta_7+beta_22;

delta=pi/2-beta_8+beta_9;

h_i=R*cos(delta);
x_i=R*(1-sin(delta));

l_CCD=(h_CCD_max+h_i)/tan(beta_9)-x_i; %%m


