a=rand(1,1);
b=rand(1,1);

for i=1:6
	aa=tic;
	for j=1:300
		[A B]=Mie_ab_mex(complex(a*i,0),complex(b*i,0));
	end
	czas=toc(aa);
	bb=tic;
	for j=1:300
		[C D]=Mie_ab_SM(a*i,b*i);
	end
	czas2=toc(bb);
	iloraz(i)=czas2/czas;
end
mean(iloraz)
	