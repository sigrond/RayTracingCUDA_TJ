a=rand(1,1);
b=rand(1,1);

for i=1:10
	aa=tic;
	for j=1:100
		Mie_ab(a*i,b*i);
	end
	czas=toc(aa);
	bb=tic;
	for j=1:100
		Mie_ab_SM(a*i,b*i);
	end
	czas2=toc(bb);
	iloraz(i)=czas/czas2;
end
mean(iloraz)
	