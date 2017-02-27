#include<cstdio>
#include<cmath>
#include"auxillary.h"
void Calculate_m(double &m, double temp, double lambda, int liquid) {
	double T_ref=20.0;
	double A0;
	double A1;
	double A2;
	double A3;
	switch(liquid) {
		case 1: //EG
			A0 = 1.42522;
			A1 = 1984.54821;
			A2 = 104174632.0;
			m = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)));
			break;
		case 2: //2EG
			A0 = 1.4406; 
			A1 = 1984.54821;
			A2 = 104174632;
			m = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)));

			break;
		case 3: //3EG
			A0 = 1.4492;
			A1 = 1984.54821;
			A2 = 104174632;
			m = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)));
			break;
		case 4: //4EG
			A0 = 1.4511;
			A1 = 1984.54821;
			A2 = 104174632;
			m = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)));
			break;
		case 5://DMI
			A0 = 1.4664;  
			A1 = 1984.54821;
			A2 = 104174632;
			m = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)));
			break;
		case 6://DMSO
			A0 = 1.4585;  
			A1 = 8584;
			A2 = -485600000;
			m = -0.00044*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)));
			break;
		case 7: //H2O
			{
			double a0 = 0.244257733; 
			double a1 = 9.74634476e-3;
			double a2 = -3.73234996e-3; 
			double a3 = 2.68678472e-4;
			double a4 = 1.5892057e-3;
			double a5 = 2.45934259e-3;
			double a6 = 0.90070492;
			double a7 = -1.66626219e-2;
			double lambda_uv = 0.229202;
			double lambda_IR = 5.432937;
			double laser = lambda/589.0;

			double T_K = 273.15 + temp;
			double Rho;
			if( temp>0 && temp<40.0 ) {
				double b1 = -3.983035;
				double b2 = 301.797;
				double b3 = 522528.9;
				double b4 = 69.34881;
				double b5 = 999.972; //% tap water (Chappuis)
				Rho = b5 * (1.0 - sq(temp+b1)*(temp+b2)/(b3*(temp+b4)));
				Rho=Rho/1000.0;
			}
			else {
				Rho = 1.0 - (temp + 288.9414)/(508929.2*((temp + 68.12963)))*(sq(temp-3.9863));
			}
			double F = Rho * (a0 + a1*Rho + a2*T_K/273.15 + a3*sq(laser)*T_K/273.15 + a4/sq(laser) 
					+ a5/(sq(laser) - sq(lambda_uv)) + a6/(sq(laser) - sq(lambda_IR)) + a7*sq(Rho));
			m = sqrt( (2.0*F+1.0)/(1.0-F) );
			break;
			}
		case 8: // Meth
			A0=1.3195;
			A1=3053.64419;
			A2=-34163639.3011;
			A3=2.622128e+12;
			m = -0.00040*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda)))+A3/(sq(lambda)*sq(sq(lambda))) ;
			break;
		case 9: //H20 + 2EG
			{
			double a0 = 0.244257733; 
			double a1 = 9.74634476e-3;
			double a2 = -3.73234996e-3; 
			double a3 = 2.68678472e-4;
			double a4 = 1.5892057e-3;
			double a5 = 2.45934259e-3;
			double a6 = 0.90070492;
			double a7 = -1.66626219e-2;
			double lambda_uv = 0.229202;
			double lambda_IR = 5.432937;
			double laser = lambda/589.0;

			double T_K = 273.15 + temp;
			double Rho;
			if( temp>0 && temp<40.0 ) {
				double b1 = -3.983035;
				double b2 = 301.797;
				double b3 = 522528.9;
				double b4 = 69.34881;
				double b5 = 999.972; //% tap water (Chappuis)
				Rho = b5 * (1.0 - sq(temp+b1)*(temp+b2)/(b3*(temp+b4)));
				Rho=Rho/1000.0;
			}
			else {
				Rho = 1.0 - (temp + 288.9414)/(508929.2*((temp + 68.12963)))*(sq(temp-3.9863));
			}
			double F = Rho * (a0 + a1*Rho + a2*T_K/273.15 + a3*sq(laser)*T_K/273.15 + a4/sq(laser) 
					+ a5/(sq(laser) - sq(lambda_uv)) + a6/(sq(laser) - sq(lambda_IR)) + a7*sq(Rho));
			double mh = sqrt( (2.0*F+1.0)/(1.0-F) );
        A0 = 1.4406;
        A1 = 1984.54821;
        A2 = 104174632;
			double mEG = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda))) ;
			m = (mh + mEG)/2.0;
			break;
			}
		case 10: //H20 + 3EG
			{
           A0 = 1.4492;
           A1 = 1984.54821;
           A2 = 104174632;
           double m3EG = -0.00026*(temp-T_ref)+A0+A1/sq(lambda)+A2/(sq(sq(lambda))) ;
			double a0 = 0.244257733; 
			double a1 = 9.74634476e-3;
			double a2 = -3.73234996e-3; 
			double a3 = 2.68678472e-4;
			double a4 = 1.5892057e-3;
			double a5 = 2.45934259e-3;
			double a6 = 0.90070492;
			double a7 = -1.66626219e-2;
			double lambda_uv = 0.229202;
			double lambda_IR = 5.432937;
			double laser = lambda/589.0;

			double T_K = 273.15 + temp;
			double Rho;
			if( temp>0 && temp<40.0 ) {
				double b1 = -3.983035;
				double b2 = 301.797;
				double b3 = 522528.9;
				double b4 = 69.34881;
				double b5 = 999.972; //% tap water (Chappuis)
				Rho = b5 * (1.0 - sq(temp+b1)*(temp+b2)/(b3*(temp+b4)));
				Rho=Rho/1000.0;
			}
			else {
				Rho = 1.0 - (temp + 288.9414)/(508929.2*((temp + 68.12963)))*(sq(temp-3.9863));
			}
			double F = Rho * (a0 + a1*Rho + a2*T_K/273.15 + a3*sq(laser)*T_K/273.15 + a4/sq(laser) 
					+ a5/(sq(laser) - sq(lambda_uv)) + a6/(sq(laser) - sq(lambda_IR)) + a7*sq(Rho));
			double mh = sqrt( (2.0*F+1.0)/(1.0-F) );
			m = (mh+m3EG)/2.0;
			break;
			}
	}
}

