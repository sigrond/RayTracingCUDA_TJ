a=rand(1,1);
b=rand(1,1);
iloraz=zeros(1,10);
for i=1:100
	aa=tic;
	for j=1:100
		C=Mie_pt_mex_C(double(a),int32(b*i+10));
	end
	czas=toc(aa);
	bb=tic;
	for j=1:100
		A=Mie_pt_mex_C_parallel(double(a),int32(b*i+10));
	end
	czas2=toc(bb);
	iloraz(i)=czas/czas2;
	
	
end
mean(iloraz)