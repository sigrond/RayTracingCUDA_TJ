A=-1:0.001:1;

for i=10:20
	aa=tic;
	A=Mie_vector_bench(A,int32(i));
	czas1(i-9)=toc(aa);
end

for i=10:20
	bb=tic;
	[C D]=Mie_pt_vector_mex_C(A,int32(i));
	czas2(i-9)=toc(bb);
end

iloraz=czas1./czas2;
mean(iloraz)