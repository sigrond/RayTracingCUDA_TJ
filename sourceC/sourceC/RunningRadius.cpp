#include<math.h>
#include"globals.h"
#include"auxillary.h"

real find_l_CCD( real h_CCD_max, real fi, real lambda) {
	real alpha = atan(fi/(2*16.86e-3));
	real B1=1.03961212;
	real B2=0.231792344;
	real B3=1.01046945;
	real C1=6.00069867e-3;
	real C2=2.00179144e-2;
	real C3=103.560653;
	real n0 = 1.0; //wsp zalamania osrodka
	real nS = sqrt( (B1*sq(lambda))/(sq(lambda)-C1) + (B2*sq(lambda))/(sq(lambda)-C2) 
			+ (B3*sq(lambda))/(sq(lambda)-C3)+1.0);
	real R=0.018705; // promien krzywizny pow. lam. soczewki [m]
	real L=0.005; // grubosc soczewki [m]
	real f_H = 1.0/( (nS-1.0)*(2.0/R-L*(nS-1.0)/(nS*sq(R))));
	real VH = L*f_H*(nS -1.0)/(R*nS);
	real f = f_H-VH+L*0.5;
	real L_TOT = 0.04257;
	real xx_1 = (2.0*f-L)/(2.0*R)+1.0;

	real beta_1 = asin( n0/nS * xx_1 *sin(alpha));
	real beta_0 = asin( xx_1 * sin(alpha));
	real beta_00 = alpha + beta_1-beta_0;
	real beta_2 = beta_0-alpha;
	real h_0 = R * sin(beta_2);
	real x_0 = R*(1.0-cos(beta_2));
	real a_0 = h_0 / tan(beta_00);
	real a_00 = a_0 -x_0+L-R;
	real h_prim = a_00 * sin(beta_00);
	real beta_3 = asin( h_prim/R);
	real beta_4 = asin( nS/n0 * sin(beta_3));
	real beta_11 = beta_4 - beta_3 -beta_00;
	real beta_3_prim = M_PI*0.5 -beta_00;
	real h_1 = R*cos( beta_3_prim - beta_3 );
	real x_1 = R* ( 1.0 - sin(beta_3_prim- beta_3));
	real b_1 = h_1 / tan(beta_11);
	real b_11 = b_1-x_1-L_TOT-R;
	real h_11 = b_11 * sin(beta_11);
	real beta_5 = asin( h_11/R);
	real beta_6 = asin( n0/nS * sin(beta_5));
	real h_22 = R * sin(beta_6);
	real gama = beta_11+beta_5;
	real beta_22 = gama - beta_6;
	real b_22 = h_22 / sin(beta_22);
	real b_22_prim = b_22+2.0*R-L;
	real h_2_prim = b_22_prim * sin(beta_22);
	real beta_7 = asin(h_2_prim/R);
	real beta_8 = asin( nS/n0 * sin(beta_7));
	real beta_9 = beta_8-beta_7+beta_22;
	real delta = M_PI * 0.5 -beta_8+beta_9;
	real h_i = R * cos(delta);
	real x_i = R* (1.0 - sin(delta));
	real l_CCD = (h_CCD_max + h_i)/tan(beta_9)-x_i;
	return l_CCD;

}
//returns SQUARE of result (inverted)
void RunningRadius(real * r, real * alpha, int alphaSize  , real h_CCD_max, real fi, real lambda) {
	//h_CCD_max=0.00277992;
	lambda*=1e-3;
	//wspolczynniki dyspersji dla szkla BK7
	real B1=1.03961212;
	real B2=0.231792344;
	real B3=1.01046945;
	real C1=6.00069867e-3;
	real C2=2.00179144e-2;
	real C3=103.560653;
	real n0 = 1.0; //wsp zalamania osrodka
	real nS = sqrt( (B1*sq(lambda))/(sq(lambda)-C1) + (B2*sq(lambda))/(sq(lambda)-C2) 
			+ (B3*sq(lambda))/(sq(lambda)-C3)+1.0);
	real R=0.018705; // promien krzywizny pow. lam. soczewki [m]
	real L=0.005; // grubosc soczewki [m]
	real f_H = 1.0/( (nS-1.0)*(2.0/R-L*(nS-1.0)/(nS*sq(R))));
	real VH = L*f_H*(nS -1.0)/(R*nS);
	real f = f_H-VH+L*0.5;
	real L_TOT = 0.04257;
	real xx_1 = (2.0*f-L)/(2.0*R)+1.0;
	
	real * alpha_prim = new real[alphaSize];
	real * beta_1 = new real[alphaSize];
	real * beta_0 = new real[alphaSize];
	real * beta_00 = new real[alphaSize];
	real * beta_2 = new real[alphaSize];
	real * h_0 = new real[alphaSize];
	real * x_0 = new real[alphaSize];
	real * a_0 = new real[alphaSize];
	real * a_00 = new real[alphaSize];
	real * h_prim = new real[alphaSize];
	real * beta_3 = new real[alphaSize];
	real * beta_4 = new real[alphaSize];
	real * beta_11 = new real[alphaSize];
	real * beta_3_prim = new real[alphaSize];
	real * h_1 = new real[alphaSize];
	real * x_1 = new real[alphaSize];
	real * b_1 = new real[alphaSize];
	real * b_11 = new real[alphaSize];
	real * h_11 = new real[alphaSize];
	real * beta_5 = new real[alphaSize];
	real * beta_6 = new real[alphaSize];
	real * h_22 = new real[alphaSize];
	real * gama = new real[alphaSize];
	real * beta_22 = new real[alphaSize];
	real * b_22 = new real[alphaSize];
	real * b_22_prim = new real[alphaSize];
	real * h_2_prim= new real[alphaSize];
	real * beta_7= new real[alphaSize];
	real * beta_8= new real[alphaSize];
	real * beta_9= new real[alphaSize];
	real * delta = new real[alphaSize];
	real * h_i = new real[alphaSize];
	real * x_i = new real[alphaSize];
	real * h_CCD_norm = new real[alphaSize];
	real l_CCD = find_l_CCD(h_CCD_max, fi, lambda);
	//l_CCD=0.023686793600964;

	for(int i=0;i<alphaSize;++i) {
		alpha_prim[i] = fabs( alpha[i] - M_PI*0.5);
		beta_1[i] = asin( n0/nS * xx_1 *sin(alpha_prim[i]));
		beta_0[i] = asin( xx_1 * sin(alpha_prim[i]));
		beta_00[i] = alpha_prim[i] + beta_1[i]-beta_0[i];
		beta_2[i] = beta_0[i]-alpha_prim[i];
		h_0[i] = R * sin(beta_2[i]);
		x_0[i] = R*(1.0-cos(beta_2[i]));
		a_0[i] = h_0[i] / tan(beta_00[i]);
		a_00[i] = a_0[i] -x_0[i]+L-R;
		h_prim[i] = a_00[i] * sin(beta_00[i]);
		beta_3[i] = asin( h_prim[i]/R);
		beta_4[i] = asin( nS/n0 * sin(beta_3[i]));
		beta_11[i] = beta_4[i] - beta_3[i] -beta_00[i];
		beta_3_prim[i] = M_PI*0.5 -beta_00[i];
		h_1[i] = R*cos( beta_3_prim[i] - beta_3[i] );
		x_1[i] = R* ( 1.0 - sin(beta_3_prim[i]- beta_3[i]));
		b_1[i] = h_1[i] / tan(beta_11[i]);
		b_11[i] = b_1[i]-x_1[i]-L_TOT-R;
		h_11[i] = b_11[i] * sin(beta_11[i]);
		beta_5[i] = asin( h_11[i]/R);
		beta_6[i] = asin( n0/nS * sin(beta_5[i]));
		h_22[i] = R * sin(beta_6[i]);
		gama[i] = beta_11[i]+beta_5[i];
		beta_22[i] = gama[i] - beta_6[i];
		b_22[i] = h_22[i] / sin(beta_22[i]);
		b_22_prim[i] = b_22[i]+2.0*R-L;
		h_2_prim[i] = b_22_prim[i] * sin(beta_22[i]);
		beta_7[i] = asin(h_2_prim[i]/R);
		beta_8[i] = asin( nS/n0 * sin(beta_7[i]));
		beta_9[i] = beta_8[i]-beta_7[i]+beta_22[i];
		delta[i] = M_PI * 0.5 -beta_8[i]+beta_9[i];
		h_i[i] = R * cos(delta[i]);
		x_i[i] = R* (1.0 - sin(delta[i]));
		h_CCD_norm[i] = (tan(beta_9[i]) * (l_CCD+x_i[i]) -h_i[i])/h_CCD_max;
		r[i] = 1.0/sq(h_CCD_norm[i] / sin(beta_9[i]));
	}
	delete [] alpha_prim;
	delete [] beta_1 ;
	delete [] beta_0;
	delete [] beta_00;
	delete [] beta_2;
	delete [] h_0;
	delete [] x_0;
	delete [] a_0;
	delete [] a_00 ;
	delete [] h_prim;
	delete [] beta_3;
	delete [] beta_4;
	delete [] beta_11;
	delete [] beta_3_prim ;
	delete [] h_1;
	delete [] x_1;
	delete [] b_1;
	delete [] b_11;
	delete [] h_11;
	delete [] beta_5;
	delete [] beta_6;
	delete [] h_22 ;
	delete [] gama ;
	delete [] beta_22;
	delete [] b_22 ;
	delete [] b_22_prim ;
	delete [] h_2_prim;
	delete [] beta_7;
	delete [] beta_8;
	delete [] beta_9;
	delete [] delta ;
	delete [] h_i ;
	delete [] x_i;
	delete [] h_CCD_norm ;
}

		




