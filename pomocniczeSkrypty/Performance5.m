a=rand(1,1);
b=rand(1,1);
iloraz=zeros(1,10);
for i=1:10
	aa=tic;
	for j=1:100
		C=Mie_ab_mex_C(complex(a*i,0),complex(b*i,0));
	end
	czas=toc(aa);
	bb=tic;
	for j=1:100
		A=Mie_ab_mex_omp(complex(a*i,0),complex(b*i,0));
	end
	czas2=toc(bb);
	iloraz(i)=czas/czas2;
	
	
end
mean(iloraz)